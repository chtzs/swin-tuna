import seaborn as sns
import matplotlib.pyplot as plt

import json

from typing import List

class TrainData:
    def __init__(self, name: str, path: List[str], color="#ffff00"):
        self.name = name
        self.path = path
        self.color = color
        self.loss_data = []
        self.eval_data = []
        
        for p in self.path:
            with open(p) as f:
                # 每一行是一个json数据
                lines = f.readlines()
            for line in lines:
                json_data = json.loads(line)
                if 'loss' in json_data:
                    self.loss_data.append(json_data)
                else:
                    self.eval_data.append(json_data)
                    
        self.loss_data = self.remove_duplicated(self.loss_data)
        self.eval_data = self.remove_duplicated(self.eval_data)

            
    def remove_duplicated(self, data: List[object]) -> List[object]:
        # 去重
        data = sorted(data, key=lambda x: x['step'])
        re = []
        for item in data:
            if len(re) > 0 and re[-1]['step'] == item['step']:
                # print(re[-1])
                # print(item)
                continue
            re.append(item)
        return re
                
    def __getitem__(self, idx: int):
        return self.loss_data[idx]
        
    def __len__(self):
        return len(self.loss_data)
        
def base_draw(data: List[TrainData], draw_func):      
    data_len = len(data[0])
    for item in data:
        assert len(item) == data_len, "Wrong data"
        
    draw_func()
    
    # 添加标题和标签
    plt.title("Title", fontweight='bold', fontsize=14)
    plt.xlabel("X Label", fontsize=12)
    plt.ylabel("Y Label", fontsize=12)

    # 添加图例
    plt.legend(loc='upper left', frameon=True, fontsize=10)

    # 设置刻度字体和范围
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 设置坐标轴样式
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)

def draw_train_loss(data: List[TrainData]):
    def inner_draw():
        for item in data:
            x = [i['step'] for i in item.loss_data]
            y = [i['loss'] for i in item.loss_data]
            sns.lineplot(x=x, y=y, color=item.color, linewidth=0.5, label=item.name)
    base_draw(data, inner_draw)
    plt.savefig("loss.pdf", format='pdf')
    
def draw_mIoU(data: List[TrainData]):
    def inner_draw():
        for item in data:
            x = [i['step'] for i in item.eval_data]
            y = [i['mIoU'] for i in item.eval_data]
            sns.lineplot(x=x, y=y, color=item.color, linewidth=2, label=item.name)
    base_draw(data, inner_draw)
    plt.savefig("mIoU.pdf", format='pdf')
    
data = [
    TrainData("TUNA", ["tmp/foodseg103/tuna.json"], color="#038355"),
    TrainData("TUNA-new", ["tmp/foodseg103/tuna_new1.json", "tmp/foodseg103/tuna_new2.json"], color="#ffc34e"),
    TrainData("Mona", ["tmp/foodseg103/mona1.json", "tmp/foodseg103/mona2.json"])
]
# draw_train_loss(data)
draw_mIoU(data)