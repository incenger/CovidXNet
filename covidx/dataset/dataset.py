from torchvision.datasets import ImageFolder
import random
class CovidxDataset(ImageFolder):
    def __init__(self,root,transform):
        super().__init__(root)
        self.covid_sample = [s for s in self.samples if s[1]==0] 
        self.normal_sample = [s for s in self.samples if s[1]==1]
        self.pneumonia_sample = [s for s in self.samples if s[1]==2]
        self.transform = transform
        self.turn = 0
    def shuffleSamples(self):
        random.shuffle(self.covid_sample)
        random.shuffle(self.normal_sample)
        random.shuffle(self.pneumonia_sample)
    def __getitem__(self, index):
        
        pos = index // 3
        self.turn%=3
        if self.turn==0:
            pos %= len(self.pneumonia_sample)
            path,target = self.pneumonia_sample[pos]
        elif self.turn ==1:
            pos %= len(self.normal_sample)
            path,target = self.normal_sample[pos]
        else:
            pos %= len(self.covid_sample)
            path,target = self.covid_sample[pos]
        self.turn +=1
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target