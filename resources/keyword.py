# Sebastian Thomas (coding at sebastianthomas dot de)

# web development
from flask import request
from flask_restful import Resource

# custom modules
from keyword_spotter import KeywordSpotter


__all__ = ['KeywordResource']


class KeywordResource(Resource):
    @staticmethod
    def post():
        audio_file = request.files['audio']
        # if not found, Flask-RESTful responds with status code 400

        try:
            predicted_keyword = KeywordSpotter.predict(audio_file)
        except FileNotFoundError:
            return {'message': 'An error occurred while trying to predict the '
                               'keyword in the file.'}, 500

        if predicted_keyword == 'unknown':
            return {'message': 'The keyword is unknown.'}, 200

        return {'keyword': predicted_keyword}, 200
