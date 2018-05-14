python test_main.py --model resnet --lr 1e-2 --optimizer adam --save_model
python test_main.py --model resnet --lr 1e-3 --optimizer adam --save_model
python test_main.py --model resnet --lr 1e-4 --optimizer adam --save_model

python test_main.py --model resnet --lr 1e-2 --optimizer sgd --save_model
python test_main.py --model resnet --lr 1e-3 --optimizer sgd --save_model
python test_main.py --model resnet --lr 1e-4 --optimizer sgd --save_model

python test_main.py --model resnet --lr 1e-2 --optimizer rmsprop --save_model
python test_main.py --model resnet --lr 1e-3 --optimizer rmsprop --save_model
python test_main.py --model resnet --lr 1e-4 --optimizer rmsprop --save_model