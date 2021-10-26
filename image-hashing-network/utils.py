import os
import cv2
from collections import defaultdict
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageCollector():
    
    def __init__(self,classes):
        self.classes=classes
        self.img_idx_to_base_idxs = defaultdict(list)
        self.actual_count=0
        self.transformed_count=0
        self.epochs=4
        
    def collect_anchor_images(self,base_dir, ext):
        
        for cl in self.classes:
            if ext not in ['.png','.jpg','.jpeg']:
                raise 

            img_path = os.path.join(base_dir, cl)
            img_fpath = glob.glob(img_path + '/*'+ext)
            yield img_fpath, cl
    
    def move_anchor_images(self,img_fpath,copy_dir, cl, IMG_SHAPE, IS_BGR=False):
        for fpath in img_fpath:
            img = cv2.imread(fpath)
            
            if not os.path.exists(os.path.join(copy_dir, cl)):
                os.makedirs(os.path.join(copy_dir, cl))
            
            img = cv2.resize(img,(IMG_SHAPE,IMG_SHAPE))
            copy_fpath = os.path.join(copy_dir, cl,str(self.actual_count)+'.png')
            self.img_idx_to_base_idxs[copy_fpath].append(copy_fpath)
            
            if IS_BGR==True:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(copy_fpath,img)
            
            self.actual_count+=1

    def create_transformed_images(self,base_dir,copy_dir, PARAMS, IMG_SHAPE, batch_size, IS_BGR=False):
    
        image_gen = ImageDataGenerator(**PARAMS)

        data_gen = image_gen.flow_from_directory(
                                                    batch_size=batch_size,
                                                    directory=base_dir,
                                                    shuffle=False,
                                                    target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='sparse'
                                                    )
        
        class_lkup = {v:k for k,v in data_gen.class_indices.items()}

        for _ in range(data_gen.n//batch_size * self.epochs):
            batch,class_idx = data_gen.next()

        #     idx = (train_data_gen.batch_index - 1) * train_data_gen.batch_size
        #     actual_file_names =  train_data_gen.filenames[idx : idx + train_data_gen.batch_size]

            idx_left = (data_gen.batch_index - 1) * batch_size
            idx_right = idx_left + data_gen.batch_size if idx_left >= 0 else None
            indices = data_gen.index_array[idx_left:idx_right]
            actual_file_names = [data_gen.filenames[i] for i in indices]

            N = batch.shape[0]
            for i in range(N):
                img = batch[i]

                cl = class_lkup[int(class_idx[i])]
                file_name = actual_file_names[i]
        #         cl = file_name.split('\\')[0]

                img = cv2.resize(img,(IMG_SHAPE,IMG_SHAPE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if not os.path.exists(os.path.join(copy_dir, cl)):
                    os.makedirs(os.path.join(copy_dir, cl))

                base_fpath = base_dir+'\\'+file_name
                copy_fpath = os.path.join('transformed', cl,str(self.transformed_count)+'.png')
                self.img_idx_to_base_idxs[base_fpath].append(copy_fpath)
                
                if IS_BGR==True:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
                cv2.imwrite(copy_fpath,img)
                self.transformed_count+=1
                