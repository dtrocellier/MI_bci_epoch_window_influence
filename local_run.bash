#python main.py -m \
#    dataset=Lee2019_MI \
#    epoch_window=w_00_30 \
#    subject=$(seq -s, 1 54)

#python main.py -m \
#    dataset=Lee2019_MI \
#    epoch_window=w_05_35 \
#    subject=$(seq -s, 1 54)

#python main.py -m \
#    dataset=Dreyer2023 \
#    epoch_window=w_00_40 \
#    subject=$(seq -s, 1 87)

python main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_05_45 \
    subject=$(seq -s, 46 87)
