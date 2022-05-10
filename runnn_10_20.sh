python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.0001 --optimizer SGD --betas1 0.8 --betas2 0.999 --momentum 0.9
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res10Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res10F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.005 --optimizer Adam --betas1 0.8 --betas2 0.99 --momentum 0.9
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res11Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res11F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.005 --optimizer Adam --betas1 0.9 --betas2 0.99 --momentum 0.85
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res12Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res12F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.01 --optimizer SGD --betas1 0.9 --betas2 0.99 --momentum 0.9
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res13Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res13F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.001 --optimizer Adam --betas1 0.85 --betas2 0.99 --momentum 0.85
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res14Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res14F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.001 --optimizer SGD --betas1 0.9 --betas2 0.99 --momentum 0.85
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res15Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res15F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.001 --optimizer Adam --betas1 0.9 --betas2 0.999 --momentum 0.95
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res16Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res16F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.01 --optimizer Adam --betas1 0.85 --betas2 0.99 --momentum 0.95
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res17Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res17F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.005 --optimizer SGD --betas1 0.8 --betas2 0.999 --momentum 0.9
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res18Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res18F.txt

python3 train.py --dataset middlebury --datapath ./ --trainlist ./filenames/mb_H.txt --testlist ./filenames/mb_Q.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --model MSNet2D --logdir my_logs --lr 0.001 --optimizer Adam --betas1 0.9 --betas2 0.99 --momentum 0.95
python3 test.py --datapath ./ --testlist ./filenames/mb_Q.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res19Q.txt
python3 test.py --datapath ./ --testlist ./filenames/mb_F.txt --loadckpt ./my_logs/best.ckpt --dataset middlebury --test_batch_size 1 --model MSNet2D > res19F.txt
