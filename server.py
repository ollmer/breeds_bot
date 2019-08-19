from tg import TgApi
from fastai.vision import *

import logging

logging.basicConfig(format='[%(asctime)s] - %(name)s - %(funcName)s - %(levelname)s - %(message)s', 
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')], level=logging.INFO)


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
    logging.info('run server')
    tg = TgApi()
    logging.info('load models')
    breeder = load_learner('model')
    imagenet = load_learner('imagenet')
    logging.info('ready')
    for m in tg.get_message():
        uid = m['chat']['id']
        photo = m.get('photo')
        if not photo:
            tg.answer('Отправьте фото котика или пса', uid)
            continue
        logging.info('photo %s', photo[0])
        raw = tg.get_file(photo[0]['file_id'])
        img = open_image(BytesIO(raw))
        pred_class,_,_ = imagenet.predict(img)
        what = str(pred_class)
        cat_or_dog = what in cd
        logging.info('imagenet %s', pred_class)
        if not cat_or_dog:
            tg.answer('Это не кот и не собака. Похоже на ' + what, uid)
        else:
            breed,pred_idx,outputs = breeder.predict(img)
            breed = str(breed).replace('_', ' ').capitalize()
            confidence = float(outputs[pred_idx])
            logging.info('prediction imagenet %s, main %s, confidence %.3f', pred_class, breed, confidence)
            answer = breed + ' (%.2f)' % confidence
            tg.answer(answer, uid)