--frac 客户比例 --lr 学习率 --beta 超参数  --gamma_star 超参数 --nl 噪声等级 --ls 全局抽样数 --local_ep 本地更新的epoch数 --epsilon ε值 --an 是否是自适应 --iid 是否是独立同分布 --save 是否保存一些参数


（delta值可以不管，一直默认为1e-4）

## Requirements
python>=3.6  
pytorch>=0.4

python main_0324.py --num_users 50 --frac 1.0 --local_ep 8 --lr 0.001 --beta 1.1 --lcf 4.0 --nl 22.0 --epsilon 0.5 --iid --an --optimizer Momentum


python fed_repeat.py --num_users 50 --frac 1.0 --local_ep 1 --lr 0.002 --beta 1.2 --gamma_star 1.1 --lcf 4.0 --nl 22.0 --epsilon 0.5 --iid --an

nohup python fed_repeat.py --num_users 50 --frac 0.2 --local_ep 1 --lr 0.001 --beta 1.2 --gamma_star 1.2 --lcf 4.0 --nl 22.0 --epsilon 0.5 --iid --an >>mylog.out 2>&1 &

nohup python fed_store.py --num_users 50 --frac 1.0 --local_ep 1 --lr 0.001 --beta 1.2 --gamma_star 1.1 --lcf 4.0 --nl 22.0 --epsilon 0.5 --iid --an >>mylog.out 2>&1 &

python fed_fin.py --dataset cifar --num_channels 3 --model cnn --rounds 150 --gpu 0 --nl 22.0 --epsilon 2.0 --delta 1e-4 --num_users 100 --frac 1.0 --an --iid --lr 0.001 --local_ep 1 #自适应独立同分布（ε=2）

python fed_anFalse.py --dataset mnist --num_channels 1 --model dppca --rounds 5000 --gpu -1 --nl 22.0 --epsilon 0.5 --delta 1e-4 --num_users 100 --frac 1.0 --iid --lr 0.01 --local_ep 1 #非自适应

python fed_fin.py --dataset cifar --num_channels 3 --model cnn --rounds 150 --gpu 0 --nl 22.0 --epsilon 0.5 --delta 1e-4 --num_users 10 --frac 1.0 --lr 0.001 --local_ep 1 --iid --an #自适应非独立同分布（ε=2）

python main_fed_1_6.py --dataset mnist --num_channels 1 --model dppca --epochs 250 --gpu -1 --nl 30 --epsilon 0.5 --delta 1e-4 --num_users 50 --frac 1.0 --an --iid --lr 0.001 #自适应独立同分布（ε=8）

python main_fed_1_6.py -dataset mnist --num_channels 1 --model dppca --epochs 250 --gpu -1 --nl 3.0 --epsilon 8.0 --delta 1e-4 --num_users 50 --frac 1.0 --an --lr 0.001 #自适应非独立同分布（ε=8）

python main_fed_1_10.py --dataset mnist --num_channels 1 --model dppca --epochs 250 --gpu -1 --nl 3.0 --epsilon 0.5 --delta 1e-4 --num_users 50 --frac 1.0 --an --iid --lr 0.001 --gamma_star 0.8

## 参数对模型影响

### 训练率
python main_fed_1_6.py --dataset mnist --num_channels 1 --model dppca --epochs 250 --gpu -1 --nl 3.0 --epsilon 0.5 --delta 1e-4 --num_users 50 --frac 1.0 --an --iid --lr 0.001 








