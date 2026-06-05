N_SUBJECTS=$(python get_subject_list.py dataset=Dreyer2023)

sbatch --array=1-${N_SUBJECTS} run.sbatch \
    dataset=Dreyer2023 \
    model=Deep4Net