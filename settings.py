import logging


def initialize_setting(current_experiment):
    """
    initialize the general setting of each experiment
    :param current_experiment: experiment name or number
    :return:
    """
    setting = dict()

    setting["current_experiment"] = current_experiment
    setting["voxel_size"] = [1, 1, 1]

    # load data settings
    setting["data"] = dict()
    setting["data"]["DIR-Lab"] = load_data_setting(selected_data="DIR-Lab")
    setting["data"]["SBU"] = load_data_setting(selected_data="SBU")

    return setting

def load_data_setting(selected_data):
    """
    load the general setting of selected data
    :param selected_data: it must be in ["DIR-Lab_4D", "SBU"]
    :return:
    """

    data_setting = dict()
    if selected_data == "DIR-Lab":
        data_setting["root"] = "./data"
        data_setting["ext"] = ".mha"  # the extension of each volume of data
        data_setting["phases"] = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
        data_setting["default_pixel_value"] = -2048  # the pixel value when a transformed pixel is outside the image
        data_setting["voxel_size"] = [1, 1, 1]
        data_setting["case_numbers_list"] = [i for i in range(1, 11)] # number of cases in the dataset
    elif selected_data == "SBU":
        data_setting["root"] = "./data"
        data_setting["ext"] = ".mha"  # the extension of each volume of data
        data_setting["phases"] = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
        data_setting["default_pixel_value"] = -2048  # the pixel value when a transformed pixel is outside the image
        data_setting["voxel_size"] = [1, 1, 1]
        data_setting["case_numbers_list"] = [i for i in range(1, 7)]  # number of cases in the dataset
    else:
        logging.warning('warning: -------- selected_data not found')
    return data_setting