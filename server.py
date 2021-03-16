# Sebastian Thomas (datascience at sebastianthomas dot de)

# web development
from flask import Flask
from flask_restful import Api

# custom modules
from resources import KeywordResource


def create_app():
    app = Flask(__name__)

    api = Api(app)
    api.add_resource(KeywordResource, '/keyword')

    return app


if __name__ == '__main__':
    create_app().run(debug=False)
