import pandas as pd
import multiprocessing.dummy as mp
from nltk.corpus import stopwords
import io
from tqdm import tqdm
from itertools import islice
import numpy as np


def load_vectors(fname, limit):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(islice(fin, limit), total=limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


vecs = load_vectors('./lan_model/crawl-300d-2M.vec', 100000)


class Document:
    def __init__(self, json_data: dict):
        self.title: str = json_data['title']
        self.text: str = json_data['plot']
        self.year: int = json_data['year']
        self.genre: str = json_data['genre']
        self.origin: str = json_data['origin']
        self.cast: str = json_data['cast']
        self.director: str = json_data['director']
        self.full_content = '. '.join(list(map(str, list(json_data.values()))))

    def format(self) -> list:
        # возвращает пару тайтл-текст, отформатированную под запрос
        return ['{title} ({year})'.format(title=self.title, year=self.year),
                self.text[0:100] + ' ...']

    def count_word_in_content(self, word) -> int:
        '''
        Counts how many times the word is presented in the document content
        :param word: a counted word
        :return: number of entries
        '''

        return self.full_content.count(word)

    def count_word_in_title(self, word) -> int:
        '''
        Counts how many times the word is presented in the document title
        :param word: a counted word
        :return: number of entries
        '''

        return self.title.count(word)

    def calculate_word_relevance_for_title(self, word):
        '''
        Counts the cumulative distance between the word and the doc's title words using word2vec model
        :param word: the basic word
        :return: relevance (it is not normed and is an ordinal value designed to compare different words' relevance
        to the same document
        '''
        rel = 0
        for title_word in self.title:
            rel += np.linalg.norm(vecs[word] - vecs[title_word])
        return rel

    def calculate_word_relevance_for_all(self, word):
        '''
                Counts the cumulative distance between the word and the doc's words using word2vec model
                :param word: the basic word
                :return: relevance (it is not normed and is an ordinal value designed to compare different words' relevance
                to the same document
                '''
        rel = 0
        for doc_word in self.full_content:
            rel += np.linalg.norm(vecs[word] - vecs[doc_word])
        return rel


USED_CORES_NUMBER = 8

search_index: dict = {}
films = pd.read_csv('AI/films.csv')
films.drop(columns='Wiki Page', inplace=True)

sw_eng = set(stopwords.words('english'))


def process_line(ind):
    # для реализации многопоточности вынес функцию обработки отдельного документа-строки в датасете
    global search_index
    
    if len(search_index) % 1000 > 990:
        print("In process")
    year = films.loc[ind]['Release Year']
    title = films.loc[ind]['Title']
    origin = films.loc[ind]['Origin/Ethnicity']
    director = films.loc[ind]['Director']
    cast = films.loc[ind]['Cast']
    genre = films.loc[ind]['Genre']
    plot = films.loc[ind]['Plot']

    film = {
        'year': int(year) if (str(year).lower() != 'unknown') and (str(year).lower() != 'nan') else None,
        'title': str(title) if (str(title).lower() != 'unknown') and (str(title).lower() != 'nan') else None,
        'origin': str(origin) if (str(origin).lower() != 'unknown') and (str(origin).lower() != 'nan') else None,
        'director': str(director) if (str(director).lower() != 'unknown') and (
                str(director).lower() != 'nan') else None,
        'cast': str(cast) if (str(cast).lower() != 'unknown') and (str(cast).lower() != 'nan') else None,
        'genre': str(genre) if (str(genre).lower() != 'unknown') and (str(genre).lower() != 'nan') else None,
        'plot': str(plot) if (str(plot).lower() != 'unknown') and (str(plot).lower() != 'nan') else None,
    }

    full_text = ' '.join(map(str, film.values())).lower()

    for key_word in full_text.split():
        if key_word not in search_index:
            search_index[key_word] = set()
        search_index[key_word].add(Document(film))


def build_index():
    # считывает сырые данные и строит индекс
    # для ускорения в параллельных потоках обрабатывает каждый документ
    global search_index
    pool = mp.Pool(USED_CORES_NUMBER)
    search_index = mp.Manager().dict()
    print("Building index")
    pool.map(process_line, range(len(films)))
    print("Index built successfully")


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    query = query.split()
    rel = 0
    for word in query:
        rel += 2 * document.calculate_word_relevance_for_title(word) + \
               document.calculate_word_relevance_for_all(word)
    return rel // len(query)


def retrieve(query):
    query = ' '.join(list(filter(lambda query_word: query_word not in sw_eng, query.split())))
    global search_index
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    assert len(search_index) != 0
    candidates = []
    for word in query.split():
        if word.lower() in search_index:
            for doc in search_index[word.lower()]:
                if doc not in candidates:
                    candidates.append(doc)
    return candidates[:50]
