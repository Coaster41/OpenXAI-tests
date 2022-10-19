# Utils
import torch
import numpy as np
import pickle
import csv

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.Explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Perturbation methods required for the computation of the relative stability metrics
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation


# Choose the model and the data set you wish to generate explanations for
data_loader_batch_size = 4096
data_name = 'compas' # must be one of ['heloc', 'adult', 'german', 'compas']
model_name = 'ann'    # must be one of ['lr', 'ann']

# Hyperparameters for Lime
lime_mode = 'tabular'
lime_sample_around_instance = True
lime_kernel_width = 0.75
lime_n_samples = 1000
lime_discretize_continuous = False
lime_standard_deviation = float(np.sqrt(0.03))

# Get training and test loaders
loader_train, loader_test = return_loaders(data_name=data_name,
                                        download=True,
                                        batch_size=data_loader_batch_size)

data_iter = iter(loader_test)
inputs, labels = data_iter.next()
# print("total size", data_name)
print("dataloader size of", len(loader_train), len(loader_test))
print("size of inputs/labels",len(inputs), len(labels))
# for i in range(1, len(inputs)):
#     if i%155==0:
#         print(inputs[i-1], inputs[i], inputs[i+1])
labels = labels.type(torch.int64)

# get full training data set
data_all = torch.FloatTensor(loader_train.dataset.data)
print("data_all length",len(data_all))

# Load pretrained ml model
model = LoadModel(data_name=data_name,
                ml_model=model_name,
                pretrained=True)

for _ in range(10):

    

    # You can supply your own set of hyperparameters like so:
    # param_dict_lime = dict()
    # param_dict_lime['dataset_tensor'] = data_all
    # param_dict_lime['std'] = lime_standard_deviation
    # param_dict_lime['mode'] = lime_mode
    # param_dict_lime['sample_around_instance'] = lime_sample_around_instance
    # param_dict_lime['kernel_width'] = lime_kernel_width
    # param_dict_lime['n_samples'] = lime_n_samples
    # param_dict_lime['discretize_continuous'] = lime_discretize_continuous
    # lime = Explainer(method='lime',
    #                 model=model,
    #                 dataset_tensor=data_all,
    #                 param_dict_lime=param_dict_lime)

    # lime_custom = lime.get_explanation(inputs, 
    #                                 label=labels)

    # lime_custom[0,:]
    # # print(lime_custom.size())

    # # You can also use the default hyperparameters likes so:
    # lime = Explainer(method='lime',
    #                 model=model,
    #                 dataset_tensor=data_all,
    #                 param_dict_lime=None)
    # lime_default_exp = lime.get_explanation(inputs.float(), 
    #                                         label=labels)
    # lime_default_exp[0,:]

    index = 0
    # To use a different explanation method change the method name like so
    ig = Explainer(method='ig',
                model=model,
                dataset_tensor=data_all,
                param_dict_lime=None)
    ig_default_exp = ig.get_explanation(inputs.float(), 
                                        label=labels)
    print("length of ig explainer",len(ig_default_exp))
    ig_default_exp[index,:]

    # shap = Explainer(method='shap',
    #                 model=model,
    #                 dataset_tensor=data_all,
    #                 param_dict_shap=None)
    # shap_default_exp = shap.get_explanation(inputs.float(),
    #                                         label=labels)
    # shap_default_exp[index,:]

    def generate_mask(explanation, top_k):
        mask_indices = torch.topk(explanation, top_k).indices
        mask = torch.zeros(explanation.shape) > 10
        for i in mask_indices:
            mask[i] = True
        return mask

    # Perturbation class parameters
    perturbation_mean = 0.0
    perturbation_std = 0.10
    perturbation_flip_percentage = 0.03
    if data_name == 'compas':
        feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
    # Adult feature types
    elif data_name == 'adult':
        feature_types = ['c'] * 6 + ['d'] * 7

    # Gaussian feature types
    elif data_name == 'synthetic':
        feature_types = ['c'] * 20
    # Heloc feature types
    elif data_name == 'heloc':
        feature_types = ['c'] * 23
    elif data_name == 'german':
        feature_types = pickle.load(open('./data/German_Credit_Data/german-feature-metadata.p', 'rb'))

    # Perturbation methods
    if data_name == 'german':
        # use special perturbation class
        perturbation = NewDiscrete_NormalPerturbation("tabular",
                                                    mean=perturbation_mean,
                                                    std_dev=perturbation_std,
                                                    flip_percentage=perturbation_flip_percentage)

    else:
        perturbation = NormalPerturbation("tabular",
                                        mean=perturbation_mean,
                                        std_dev=perturbation_std,
                                        flip_percentage=perturbation_flip_percentage)

    input_dict = dict()
    # index = index
    index = 0
    explainer_name = 'ig'
    for index in range(len(ig_default_exp)):
        # inputs and models
        input_dict['x'] = inputs[index].reshape(-1)
        input_dict['input_data'] = inputs
        input_dict['explainer'] = ig
        input_dict['explanation_x'] = ig_default_exp[index,:].flatten()
        input_dict['model'] = model

        # perturbation method used for the stability metric
        input_dict['perturbation'] = perturbation
        input_dict['perturb_method'] = perturbation
        input_dict['perturb_max_distance'] = 0.4
        input_dict['feature_metadata'] = feature_types
        input_dict['p_norm'] = 2
        input_dict['eval_metric'] = None

        # true label, predicted label, and masks
        input_dict['top_k'] = 3
        input_dict['y'] = labels[index].detach().item()
        input_dict['y_pred'] = torch.max(model(inputs[index].unsqueeze(0).float()), 1).indices.detach().item()
        input_dict['mask'] = generate_mask(input_dict['explanation_x'].reshape(-1), input_dict['top_k'])

        # required for the representation stability measure
        input_dict['L_map'] = model

        evaluator = Evaluator(input_dict,
                            inputs=inputs,
                            labels=labels, 
                            model=model, 
                            explainer=ig)

        if hasattr(model, 'return_ground_truth_importance'):
            rc = evaluator.evaluate(metric='RC')
            fa = evaluator.evaluate(metric='FA')
            ra = evaluator.evaluate(metric='RA')
            sa = evaluator.evaluate(metric='SA')
            sra = evaluator.evaluate(metric='SRA')
            # evaluate rank correlation
            print('RC:', rc)

            # evaluate feature agreement
            print('FA:', fa)

            # evaluate rank agreement
            print('RA:', ra)

            # evaluate sign agreement
            print('SA:', sa)

            # evaluate signed rankcorrelation
            print('SRA:', sra)

        # evaluate prediction gap on umportant features
        pgu = evaluator.evaluate(metric='PGU')
        pgi = evaluator.evaluate(metric='PGI')
        ris = evaluator.evaluate(metric='RIS')
        ros = evaluator.evaluate(metric='ROS')
        print('PGU:', pgu)

        # evaluate prediction gap on important features
        print('PGI:', pgi)

        # evaluate relative input stability
        print('RIS:', ris)

        # evaluate relative output stability
        print('ROS:', ros)

        csv_results = {}
        field_names = ['dataset', 'model', 'explainer', 'RC', 'FA', 'RA', 'SA', 'SRA', 'PGU', 'PGI', 'RIS', 'ROS']
        csv_results['dataset'] = data_name
        csv_results['model'] = model_name
        csv_results['explainer'] = explainer_name
        if hasattr(model, 'return_ground_truth_importance'):
            csv_results['RC'] = rc
            csv_results['FA'] = fa
            csv_results['RA'] = ra
            csv_results['SA'] = sa
            csv_results['SRA'] = sra
        else:
            csv_results['RC'] = ''
            csv_results['FA'] = ''
            csv_results['RA'] = ''
            csv_results['SA'] = ''
            csv_results['SRA'] = ''
        csv_results['PGU'] = pgu
        csv_results['PGI'] = pgi
        csv_results['RIS'] = ris
        csv_results['ROS'] = ros

        with open('results.csv', 'a', newline='') as csv_file:
            dict_obj = csv.DictWriter(csv_file, fieldnames=field_names)
            dict_obj.writerow(csv_results)
            csv_file.close()
    
    print('done with for loop')

    