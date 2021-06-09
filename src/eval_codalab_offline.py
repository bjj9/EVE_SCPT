from eval_codalab_basic import eval_codalab_basic

if __name__ == '__main__':
    # 1. run first round to prepare full memory
    eval_codalab_basic(output_suffix='online', skip_first_round_if_memory_is_ready=True)
    # 2. do offline evaluation when memory is ready
    eval_codalab_basic(output_suffix='offline')


