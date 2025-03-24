import os
import shutil
import random
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.mesh = None
        self.data = {}
        self.distance_list = []
        self.obj_file_path = ""

    def load_and_sample_points(self, obj_file_path, num_points=1000):
        """OBJ 파일을 불러와 포아송 디스크 샘플링으로 포인트 샘플링"""
        try:
            self.obj_file_path = obj_file_path  
            self.mesh = o3d.io.read_triangle_mesh(obj_file_path)
            self.point_cloud = self.mesh.sample_points_poisson_disk(number_of_points=num_points)
            self.points = np.asarray(self.point_cloud.points)
            
            if self.points is None or len(self.points) == 0:
                raise ValueError("샘플링된 포인트가 없습니다.")
            
            return True
        except Exception as e:
            print(f"[오류] {obj_file_path} 처리 중 오류 발생: {e}")
            folder_path = os.path.dirname(obj_file_path)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"[삭제] 오류 발생한 폴더 삭제: {folder_path}")
            return False

    def calculate_distances(self):
        """샘플링된 포인트 간의 거리를 계산"""
        if self.points is None or len(self.points) == 0:
            raise ValueError("포인트 샘플링이 진행되지 않았습니다.")

        num_points = len(self.points)
        self.distance_list = [
            np.linalg.norm(self.points[i] - self.points[j])
            for i in range(num_points) for j in range(i + 1, num_points)
        ]

        print(f"거리 데이터 계산 완료 (총 {len(self.distance_list)} 개)")
        return self.distance_list
    
    def plot_histogram(self, obj_filename, folder_name):
        """거리를 히스토그램으로 시각화 및 저장"""
        counts, bin_edges = np.histogram(self.distance_list, bins= 128, density=False)

        # 히스토그램 데이터 저장 폴더 생성
        model_folder = os.path.join(folder_name)
        os.makedirs(model_folder, exist_ok=True)

        # 히스토그램 데이터 저장
        data_save_path = os.path.join(model_folder, f"{obj_filename}.txt")
        np.savetxt(data_save_path, counts, fmt="%d", delimiter=",")

    def split_and_save_data(self, input_folder, output_folder, ratios=(0.6, 0.2, 0.2)):
        """OBJ 파일을 분할하고 거리 히스토그램을 저장"""
        assert sum(ratios) == 1.0

        os.makedirs(output_folder, exist_ok=True)
        for split in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(output_folder, split), exist_ok=True)

        for shape_folder in sorted(os.listdir(input_folder)):
            shape_path = os.path.join(input_folder, shape_folder)
            if not os.path.isdir(shape_path):
                continue
            
            variants = sorted([f for f in os.listdir(shape_path) if os.path.isdir(os.path.join(shape_path, f))])
            random.shuffle(variants)

            total = len(variants)
            train_end = int(total * ratios[0])
            val_end = train_end + int(total * ratios[1])

            splits = {
                "train": variants[:train_end],
                "validation": variants[train_end:val_end],
                "test": variants[val_end:]
            }

            for split_name, split_variants in splits.items():
                for variant in split_variants:
                    src_path = os.path.join(shape_path, variant)
                    dest_path = os.path.join(output_folder, split_name, shape_folder, variant)
                    shutil.copytree(src_path, dest_path)
                    
                    obj_files = []
                    for foldername, _, filenames in os.walk(src_path):
                        for filename in filenames:
                            if filename.lower().endswith(".obj"):
                                obj_files.append(os.path.join(foldername, filename))
                                
                    for obj_file in obj_files:
                        if self.load_and_sample_points(obj_file):
                            self.calculate_distances()
                            self.plot_histogram(os.path.basename(obj_file).replace(".obj", ""), dest_path)

        print("데이터 저장 완료!")


def main():
    input_dir = input("OBJ 파일이 포함된 최상위 폴더 경로를 입력하세요: ")
    output_dir = r"C:\Users\lee73\OneDrive\Desktop\Save_Histogram\dataset"
    
    analyzer = DataAnalyzer(output_dir)
    analyzer.split_and_save_data(input_dir, output_dir)


if __name__ == "__main__":
    main()

    # 이건 바뀐거 (test)