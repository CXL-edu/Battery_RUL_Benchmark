#
# def setup_seed(seed):
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#
#
# def build_model(model_name: str, configs: dict):
#     # 根据模型名字，导入模型
#     # 如果已经存在最优模型，就加载最优模型
#     model_dict = {
#         'MLP': 'from models.MLP import Model',
#         'GRU': 'from models.GRU import Model',
#         'BiLSTM': 'from models.BiLSTM import Model',
#         'TCN': 'from models.TCN import Model',
#         'Transformer': 'from models.Transformer import Model',
#         'ProgressiveDecompMLP': 'from models.PDMLP_auto import Model',
#     }
#     exec(model_dict[model_name])
#     model = eval("Model(configs)")
#     model.to(configs.device) if configs.use_cuda else model
#
#     # init_params(model)
#     return model