import os
import numpy as np
import pandas as pd
import logging


class WeightedSampling:
    """
    Scrappy code for oversampling or undersampling images which meet given conditions.
    """

    def __init__(self, dataframe, weighted_sampling_params, num_classes, input_size):
        self.dataframe = dataframe
        self.weighted_sampling_params = weighted_sampling_params
        self.num_classes = num_classes
        self.input_size = input_size

    def __update_human_weights(self, sample_weights):
        if self.weighted_sampling_params['humans'] != 1.0 and 'pixel_count' in self.dataframe.columns:
            print(f'Oversampling humans with weight: {self.weighted_sampling_params["humans"]}')
            weight_bool = (self.dataframe['is_human_present_in_annotations'] == True) & \
                          (self.dataframe['pixel_count'] >= self.weighted_sampling_params["human_pixels"][0]) & \
                          (self.dataframe['pixel_count'] <= self.weighted_sampling_params["human_pixels"][1])
            sample_weights[weight_bool] = self.weighted_sampling_params['humans']
        if self.weighted_sampling_params['tiny_humans'] != 1.0 and 'pixel_count' in self.dataframe.columns:
            weight_bool = (self.dataframe['is_human_present_in_annotations'] == True) & \
                          (self.dataframe['pixel_count'] <= self.weighted_sampling_params["tiny_human_pixels"])
            print(f'Sampling tiny humans ({weight_bool.sum()}) with weight: {self.weighted_sampling_params["tiny_humans"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['tiny_humans']
        if self.weighted_sampling_params['tiny_vehicles'] != 1.0 and 'vehicle_pixel_count' in self.dataframe.columns:
            weight_bool = (self.dataframe['is_vehicle_present_in_annotations'] == True) & \
                          (self.dataframe['vehicle_pixel_count'] <= self.weighted_sampling_params["tiny_vehicle_pixels"]) & \
                          (((self.dataframe['is_human_present_in_annotations'] == True) & \
                            (self.dataframe['pixel_count'] <= self.weighted_sampling_params["tiny_human_pixels"])) | \
                           (self.dataframe['is_human_present_in_annotations'] == False))
            print(f'Sampling tiny vehicles ({weight_bool.sum()}) with weight: {self.weighted_sampling_params["tiny_vehicles"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['tiny_vehicles']

    def __update_bird_weights(self, sample_weights):
        if self.weighted_sampling_params['birds'] != 1.0 and self.num_classes > 7:
            print(f'Oversampling birds with weight: {self.weighted_sampling_params["birds"]}')
            sample_weights[self.dataframe['label_map'].str.contains('Birds') &
                           (self.dataframe['state'] == 'Louisiana')] = self.weighted_sampling_params['birds']

    def __update_occluded_human_weights(self, sample_weights):
        if self.weighted_sampling_params['occluded_humans'] != 1.0 and 'occluded' in self.dataframe.columns:
            weight_bool = (self.dataframe['camera_location'] == 'rear-left') & \
                          (self.dataframe['occluded'] == True) & \
                          (self.dataframe['pixel_count'] >= self.weighted_sampling_params["occluded_human_pixels"][0]) & \
                          (self.dataframe['pixel_count'] <= self.weighted_sampling_params["occluded_human_pixels"][1]) & \
                          (self.dataframe['robot_name'].str.startswith('unk') |
                           self.dataframe['robot_name'].str.startswith('loamy'))
            print(f'Oversampling humans occluded by implement ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["occluded_humans"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['occluded_humans']

    def __update_reverse_human_weights(self, sample_weights):
        if self.weighted_sampling_params['reverse_humans'] != 1.0 and 'max_row' in self.dataframe.columns:
            weight_bool = (self.dataframe['camera_location'] == 'rear-left') & \
                          (self.dataframe['max_row'] < (self.input_size[0] // 2)) & \
                          (self.dataframe['pixel_count'] >= self.weighted_sampling_params["reverse_human_pixels"][0]) & \
                          (self.dataframe['pixel_count'] <= self.weighted_sampling_params["reverse_human_pixels"][1]) & \
                          (self.dataframe['robot_name'].str.startswith('unk') |
                           self.dataframe['robot_name'].str.startswith('loamy'))
            print(f'Oversampling humans behind implement images ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["reverse_humans"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['reverse_humans']

    def __update_triangle_human_weights(self, sample_weights):
        if self.weighted_sampling_params['triangle_humans'] != 1.0 and 'pixel_count_in_triangles' in self.dataframe.columns:
            weight_bool = (self.dataframe['camera_location'] == 'rear-left') & \
                          (self.dataframe['pixel_count_in_triangles'] >= self.weighted_sampling_params["triangle_human_pixels"][0]) & \
                          (self.dataframe['pixel_count_in_triangles'] <= self.weighted_sampling_params["triangle_human_pixels"][1]) & \
                          (self.dataframe['robot_name'].str.startswith('unk') |
                           self.dataframe['robot_name'].str.startswith('loamy'))
            print(f'Oversampling humans in triangles ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["triangle_humans"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['triangle_humans']

    def __update_day_vehicle_weights(self, sample_weights):
        if self.weighted_sampling_params['day_vehicles'] != 1.0 and 'vehicle_pixel_count' in self.dataframe.columns:
            weight_bool = (self.dataframe['operation_time'] == 'daytime') & \
                          (self.dataframe['is_vehicle_present_in_annotations'] == True) & \
                          (self.dataframe['vehicle_pixel_count'] >= self.weighted_sampling_params["day_vehicle_pixels"][0]) & \
                          (self.dataframe['vehicle_pixel_count'] <= self.weighted_sampling_params["day_vehicle_pixels"][1])
            print(f'Oversampling vehicles in day time ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["day_vehicles"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['day_vehicles']

    def __update_night_vehicle_weights(self, sample_weights):
        if self.weighted_sampling_params['night_vehicles'] != 1.0 and 'vehicle_pixel_count' in self.dataframe.columns:
            weight_bool = (self.dataframe['operation_time'] != 'daytime') & \
                          (self.dataframe['is_vehicle_present_in_annotations'] == True) & \
                          (self.dataframe['vehicle_pixel_count'] >= self.weighted_sampling_params["night_vehicle_pixels"][0]) & \
                          (self.dataframe['vehicle_pixel_count'] <= self.weighted_sampling_params["night_vehicle_pixels"][1])
            print(f'Oversampling vehicles in dawn/dusk/night time ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["night_vehicles"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['night_vehicles']

    def __update_airborne_debris(self, sample_weights):
        if self.weighted_sampling_params.get('airborne_debris', 1.0) != 1.0 and \
                'Airborne-debris' in self.dataframe.columns:
            weight_bool = (self.dataframe['Airborne-debris'] >= self.weighted_sampling_params["airborne_debris_pixels"][0]) & \
                          (self.dataframe['Airborne-debris'] <= self.weighted_sampling_params["airborne_debris_pixels"][1])
            print(f'Oversampling airborne debris images ({weight_bool.sum()}) with weight: '
                         f'{self.weighted_sampling_params["airborne_debris"]}')
            sample_weights[weight_bool] = self.weighted_sampling_params['airborne_debris']

    def update_dataframe(self):
        # # Return if there is no weighted sampling
        # use_weighted_sampling = False
        # for v in self.weighted_sampling_params.values():
        #     if v != 1.0:
        #         use_weighted_sampling = True
        #         break
        # if not use_weighted_sampling:
        #     return

        sample_weights = np.ones(len(self.dataframe))
        ori_sum = sample_weights.sum()
        self.__update_human_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)
        self.__update_bird_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)

        self.__update_day_vehicle_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)
        self.__update_night_vehicle_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)
        self.__update_airborne_debris(sample_weights)
        print(sample_weights.sum() - ori_sum)

        self.__update_occluded_human_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)
        self.__update_reverse_human_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)
        self.__update_triangle_human_weights(sample_weights)
        print(sample_weights.sum() - ori_sum)

        # create a new column in dataframe with weights for weighted sampling
        self.dataframe.loc[:, 'sample_weights'] = sample_weights


if __name__ == '__main__':
    weighted_sampling_params = {"birds": 1.0,
                            "tiny_humans": 0.0, "tiny_human_pixels": 30,
                            "tiny_vehicles": 0.0, "tiny_vehicle_pixels": 100,
                            "humans": 1.0, "human_pixels": [100, 5000],
                            "occluded_humans": 5.0, "occluded_human_pixels": [100, 2000],
                            "reverse_humans": 5.0, "reverse_human_pixels": [50, 2000],
                            "triangle_humans": 5.0, "triangle_human_pixels": [50, 2000],
                            "day_vehicles":2.0, "day_vehicle_pixels": [3000, 100000],
                            "night_vehicles":5.0, "night_vehicle_pixels": [3000, 100000],
                            "airborne_debris": 5.0, "airborne_debris_pixels": [100, 10000]}
    df = pd.read_csv('/data/jupiter/li.yu/data/Jupiter_train_v5_11/trainrd05_humanaug.csv', low_memory=False)
    print(df.shape)

    ws = WeightedSampling(df, weighted_sampling_params, 4, [512, 1024])
    ws.update_dataframe()

    # # Modify training csvs
    # # label_df = pd.read_csv('/data/jupiter/li.yu/exps/driveable_terrain_model/v53_4cls_tversky001_imgaug_60p_0209/Jupiter_train_v5_11/output.csv', low_memory=False)
    # # print('label df', label_df.shape)
    # head_lights_df = pd.read_csv('/data/jupiter/li.yu/data/Jupiter_train_v5_11/head_lights.csv')
    # print('head lights df', head_lights_df.shape)
    # csvs = ["epoch0_5_30_focal025_master_annotations.csv", "epoch0_5_30_focal05_master_annotations.csv", 
    #         "trainrd025_humanaug.csv", "trainrd05_humanaug.csv"]
    # for csv in csvs:
    #     csv_path = os.path.join('/data/jupiter/li.yu/data/Jupiter_train_v5_11/', csv)
    #     df = pd.read_csv(csv_path, low_memory=False)
    #     print(df.shape)
    #     # df = df.merge(label_df, on='id')
    #     df['vehicle_head_light'] = df.id.isin(head_lights_df.id)
    #     print(df.shape)
    #     df.to_csv(csv_path, index=False)