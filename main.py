from util import get_argparser, set_seed
from main_es_eval import main_es_eval
from main_exp import main_exp
from main_test import main_test


if __name__ == "__main__":
    args = get_argparser().parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        main_es_eval(args)
    elif args.mode == "exp":
        main_exp(args)
    elif args.mode == "test":
        main_test(args)
