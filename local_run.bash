#!/bin/bash

mkdir -p logs

uv run python main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_00_40 \
    model=Deep4Net \
    subject=$(seq -s, 1 87) \
    > logs/lee_w_00_40.out \
    2> logs/lee_w_00_40.err &

uv run python main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_05_45 \
    model=Deep4Net \
    subject=$(seq -s, 1 87)
    > logs/lee_w_05_45.out \
    2> logs/lee_w_05_45.err &

uv run python -u main.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_00_30 \
    model=Deep4Net \
    subject=$(seq -s, 1 54)
    > logs/lee_w_00_30.out \
    2> logs/lee_w_00_30.err &

uv run python -u main.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_05_35 \
    model=Deep4Net \
    subject=$(seq -s, 1 54)
    > logs/lee_w_05_35.out \
    2> logs/lee_w_05_35.err &
