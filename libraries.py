pretrained_lst = [
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'InceptionResNetV2',
    'InceptionV3',
    'MobileNet',
    'MobileNetV2',
    'NASNetLarge',
    'NASNetMobile',
    'ResNet101',
    'ResNet101V2',
    'ResNet152',
    'ResNet152V2',
    'ResNet50',
    'ResNet50V2',
    'VGG16',
    'VGG19',
    'Xception'
]

optimizer_lst = [
    'Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSprop',
    'SGD'
]


test_reviews = [
    ['I love this phone', 
     'I hate spaghetti', 
     'Everything was cold',
     'Everything was hot exactly as I wanted', 
     'Everything was green', 
     'the host seated us immediately',
     'they gave us free chocolate cake', 
     'not sure about the wilted flowers on the table',
     'only works when I stand on tippy toes', 
     'does not work when I stand on my head'],
    ['I loved this movie',
     'that was the worst movie I\'ve ever seen',
     'too much violence even for a Bond film',
     'a captivating recounting of a cherished myth',
     'I saw this movie yesterday and I was feeling low to start with, \
      but it was such a wonderful movie that it lifted my spirits and \
      brightened my day, you can\'t go wrong with a movie with Whoopi Goldberg in it.',
     'I don\'t understand why it received an oscar recommendation for \
       best movie, it was long and boring',
     'the scenery was magnificent, the CGI of the dogs was so realistic I\
      thought they were played by real dogs even though they talked!',
     'The ending was so sad and yet so uplifting at the same time. I\'m \
      looking for an excuse to see it again'
     'I had expected so much more from a movie made by the director \
      who made my most favorite movie ever, I was very disappointed in \
      the tedious story',
     'I wish I could watch this movie every day for the rest of my life']
]



tfds_data = ['glue/sst2']
data = {'reviews': 'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P',
        'song': 'https://drive.google.com/uc?id=1LiJFZd41ofrWoBtW-pMYsfz1w8Ny0Bj8'}