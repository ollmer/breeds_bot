import requests
from fastai.vision import *

dogs = ['chihuahua',  'japanese spaniel',  'maltese dog',  'pekinese',  'shih-tzu',  'blenheim spaniel',  'papillon',  'toy terrier',  'rhodesian ridgeback',  
'afghan hound',  'basset',  'beagle',  'bloodhound',  'bluetick',  'black-and-tan coonhound',  'walker hound',  'english foxhound',  'redbone',  'borzoi',  
'irish wolfhound',  'italian greyhound',  'whippet',  'ibizan hound',  'norwegian elkhound',  'otterhound',  'saluki',  'scottish deerhound',  'weimaraner',  
'staffordshire bullterrier',  'american staffordshire terrier',  'bedlington terrier',  'border terrier',  'kerry blue terrier',  'irish terrier',  'norfolk terrier',  
'norwich terrier',  'yorkshire terrier',  'wire-haired fox terrier',  'lakeland terrier',  'sealyham terrier',  'airedale',  'cairn',  'australian terrier',  
'dandie dinmont',  'boston bull',  'miniature schnauzer',  'giant schnauzer',  'standard schnauzer',  'scotch terrier',  'tibetan terrier',  'silky terrier',  
'soft-coated wheaten terrier',  'west highland white terrier',  'lhasa',  'flat-coated retriever',  'curly-coated retriever',  'golden retriever',  
'labrador retriever',  'chesapeake bay retriever',  'german short-haired pointer',  'vizsla',  'english setter',  'irish setter',  'gordon setter',  
'brittany spaniel',  'clumber',  'english springer',  'welsh springer spaniel',  'cocker spaniel',  'sussex spaniel',  'irish water spaniel',  'kuvasz',  
'schipperke',  'groenendael',  'malinois',  'briard',  'kelpie',  'komondor',  'old english sheepdog',  'shetland sheepdog',  'collie',  'border collie',  
'bouvier des flandres',  'rottweiler',  'german shepherd',  'doberman',  'miniature pinscher',  'greater swiss mountain dog',  'bernese mountain dog',  
'appenzeller',  'entlebucher',  'boxer',  'bull mastiff',  'tibetan mastiff',  'french bulldog',  'great dane',  'saint bernard',  'eskimo dog',  'malamute',  
'siberian husky',  'dalmatian',  'affenpinscher',  'basenji',  'pug',  'leonberg',  'newfoundland',  'great pyrenees',  'samoyed',  'pomeranian',  'chow',  
'keeshond',  'brabancon griffon',  'pembroke',  'cardigan',  'toy poodle',  'miniature poodle',  'standard poodle',  'mexican hairless']

cats = ['tabby',
 'tiger cat',
 'persian cat',
 'siamese cat',
 'egyptian cat',
 'cougar',
 'lynx',
 'leopard',
 'snow leopard',
 'jaguar',
 'lion',
 'tiger',
 'cheetah','madagascar cat']

cd = cats+dogs

if __name__ == '__main__':
    print('run server')
    with open('token.txt') as f:
        token = f.read().strip()
    base_url = 'https://api.telegram.org/bot%s/' % token
    print('base_url', base_url)
    print('load models')
    breeder = load_learner('model')
    imagenet = load_learner('imagenet')
    print('loaded')
    start_update = requests.get(base_url + 'getUpdates').json()['result']
    last_id = start_update[-1]['update_id'] if start_update else 0
    print('ready')
    while True:
        updates = start_update = requests.get(base_url + 'getUpdates?offset=%d' % (last_id+1)).json()['result']
        for u in updates:
            try:
                print('upd', u)
                last_id = u['update_id']
                uid = u['message']['chat']['id']
                photo = u['message'].get('photo')
                if not photo:
                    send = base_url + 'sendMessage?chat_id=%d&text=%s' % (uid, 'Отправьте фото котика или пса')
                    requests.get(send)
                    continue
                fid = photo[0]['file_id']
                print('photo', photo[0])
                file_path = requests.get(base_url + 'getFile?file_id=' + fid).json()['result']['file_path']
                flink = 'https://api.telegram.org/file/bot' + token + '/' + file_path
                print('pic', flink)
                raw = requests.get(flink).content
                img = open_image(BytesIO(raw))
                pred_class,_,_ = imagenet.predict(img)
                what = str(pred_class)
                cat_or_dog = what in cd
                print('imagenet', pred_class)
                if not cat_or_dog:
                    answer = 'Это не кот и не собака. Похоже на ' + what
                    send = base_url + 'sendMessage?chat_id=%d&text=%s' % (uid, answer)
                    requests.get(send)
                else:
                    breed,pred_idx,outputs = breeder.predict(img)
                    confidence = float(outputs[pred_idx])
                    print('pred', pred_class, breed, confidence)
                    answer = str(breed).replace('_', ' ').capitalize() + ' (%.2f)' % confidence
                    send = base_url + 'sendMessage?chat_id=%d&text=%s' % (uid, answer)
                    requests.get(send)
                
            except Exception as e:
                print('ERROR', e)
