import requests

class TgApi:
    def __init__(self, fname='token.txt'):
        with open('token.txt') as f:
            self._token = f.read().strip()
        self._base_url = 'https://api.telegram.org/bot%s/' % self._token
        self._send_tpl = self._base_url + 'sendMessage?chat_id=%d&text=%s'
        self._file_tpl = 'https://api.telegram.org/file/bot' + self._token + '/'

    def get_message(self):
        start_update = requests.get(self._base_url + 'getUpdates').json()['result']
        last_id = start_update[-1]['update_id'] if start_update else 0
        while True:
            updates = start_update = requests.get(self._base_url + 'getUpdates?offset=%d' % (last_id+1)).json()['result']
            for u in updates:
                try:
                    print('upd', u)
                    last_id = u['update_id']
                    yield u['message']
                except Exception as e:
                    print('ERROR', e)

    def answer(self, text, uid):
        requests.get(self._send_tpl % (uid, text))

    def get_file(self, file_id):
        file_path = requests.get(self._base_url + 'getFile?file_id=' + file_id).json()['result']['file_path']
        flink = self._file_tpl + file_path
        print('file link', flink)
        return requests.get(flink).content