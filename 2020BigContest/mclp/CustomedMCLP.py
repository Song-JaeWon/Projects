import pandas as pd
import numpy as np

import os

os.chdir('../')

def MCLP(dist_file, population_file, convenience_file, N=4, Ages='3059', distance=3000, w1=.7, w2=.3,
         covid_weight=[.2, .2, .2, .2, .2]):
    HDONG_ORDER = convenience_file.index.tolist()

    def double_power_distance_weight(df=dist_file, distance=distance):
        shape = df.shape
        names = df.columns.tolist()
        flatten_values = np.concatenate(df.values)
        weights = np.array([(1 - (dist / distance) ** 2) if dist < distance else 0 for dist in flatten_values])
        weights_df = pd.DataFrame(weights.reshape(shape), columns=names)
        weights_df.index = names
        return weights_df

    def update_weight_matrix(hdong=None, weight_matrix=None, HDONG_ORDER=HDONG_ORDER):
        shape = (34, 34)
        if not isinstance(weight_matrix, np.ndarray):
            weight_matrix = np.diag(np.ones((34,)))
            return weight_matrix
        else:
            if hdong not in HDONG_ORDER:
                raise ValueError('hdong must be in HDONG_ORDER')

            updated_weight_matrix = weight_matrix * (1 - double_power_distance_weight().loc[hdong].values)
            # print(np.diagonal(updated_weight_matrix))
            if updated_weight_matrix.shape == shape:
                return updated_weight_matrix
            else:
                raise ValueError(f'updated_weight_matrix\'s shape : {updated_weight_matrix.shape} is not {shape}')

    ### C
    # 교통편의성 matrix 생성
    C = np.asmatrix(convenience_file)

    ### W
    # Weighted Maxtrix 초기화(생성) - Identity Matrix
    W = update_weight_matrix()

    tmp_df = pd.DataFrame(HDONG_ORDER, columns=['HDONG_NM'])
    result = []
    class_num = population_file.Covid_class.nunique()

    for i in range(N):
        result_mat = np.zeros((34, 1))
        for covid_idx, covid_class in enumerate(population_file.Covid_class.unique()):
            ### L
            # 생활인구 - 코로나 정도 및 평일/주말로 구분
            # matrix연산 전 행정동의 순서가 제대로 되어있는지 확인 후 생활인구 Matrix 생성
            living_population_weekday = population_file.loc[(population_file.Covid_class == covid_class) &
                                                            (population_file.AGE == Ages) &
                                                            (population_file.dayofweek == 0), ['HDNG_NM', 'FLOW']]

            living_population_weekend = population_file.loc[(population_file.Covid_class == covid_class) &
                                                            (population_file.AGE == Ages) &
                                                            (population_file.dayofweek == 1), ['HDNG_NM', 'FLOW']]

            if living_population_weekday.HDNG_NM.tolist() == HDONG_ORDER:
                L_1 = np.asmatrix(living_population_weekday.set_index('HDNG_NM'))
                L_1_shape = L_1.shape
            else:
                raise ValueError('[평일]행정동의 순서가 맞지 않습니다')

            if living_population_weekend.HDNG_NM.tolist() == HDONG_ORDER:
                L_2 = np.asmatrix(living_population_weekend.set_index('HDNG_NM'))
                L_2_shape = L_2.shape
            else:
                raise ValueError('[주말]행정동의 순서가 맞지 않습니다')

            if i == 0:
                result_mat += covid_weight[covid_idx] * (w1 * (C * W * L_1) + w2 * (C * W * L_2))
                if covid_idx == (class_num - 1):
                    hdong = HDONG_ORDER[np.argmax(result_mat)]
                    result.append(hdong)
            else:
                if covid_idx == 0:
                    W = update_weight_matrix(hdong=hdong, weight_matrix=W)

                result_mat += covid_weight[covid_idx] * (w1 * (C * W * L_1) + w2 * (C * W * L_2))
                if covid_idx == (class_num - 1):
                    hdong = HDONG_ORDER[np.argmax(result_mat)]
                    result.append(hdong)

            if covid_idx == (class_num - 1):
                tmp_df[f'result_{i}'] = result_mat

    return result, tmp_df
    #return result

if __name__ == '__main__':
    # PATH 설정
    root_path = os.getcwd()

    data_folder_path = os.path.join(root_path, 'data')
    original_file_path = os.path.join(data_folder_path, 'original_data')
    original_raw_file_path = os.path.join(original_file_path, 'raw_data')
    original_processed_file_path = os.path.join(original_file_path, 'processed_data')

    raw_file_folders = os.listdir(original_raw_file_path)

    external_file_path = os.path.join(data_folder_path, 'external_data')
    external_raw_file_path = os.path.join(external_file_path, 'raw_data')
    external_processed_file_path = os.path.join(external_file_path, 'processed_data')

    # Load Data
    distance_file = [pd.read_csv(os.path.join(external_processed_file_path, file), index_col=[0]) for file in
                     os.listdir(external_processed_file_path) if file.startswith('distance')]

    real_dist = distance_file[0]

    living_population = pd.read_csv(os.path.join(external_processed_file_path, 'CTGG_HDNG_FLOW.csv'), encoding='cp949')
    convenience_index = pd.read_csv(os.path.join(external_processed_file_path, 'conv_index_df.csv'), index_col=[0])

    for n in range(4,8):
        res, process = MCLP(real_dist, living_population, convenience_index, N=n)
        print(f'자판기 설치 행정동 :{res}')
        print(process)