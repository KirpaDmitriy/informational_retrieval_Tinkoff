from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time
from multiprocessing import Manager

app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)
    scored = [(doc, score(query, doc)) for doc in documents]
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [doc.format()+['%.2f' % scr] for doc, scr in scored]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Синемарозыскъ',
        results=results
    )


if __name__ == '__main__':
    build_index()
    app.run(debug=True, host='0.0.0.0', port=80)
