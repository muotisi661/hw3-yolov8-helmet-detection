import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
from PIL import Image

class LabelImgSE:
    def __init__(self, img_dir, cls_file, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        with open(cls_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg','.jpeg'))])
        self.idx = 0
        self.rect = None
        self.start_xy = None
        self.current_boxes = []  # newly drawn person boxes this session
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('labelImgSE - person补标工具')
        self.load_image()
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.print_help()
        plt.show()

    def print_help(self):
        print('=' * 50)
        print('  labelImgSE - person类补标工具')
        print('  左键拖拽画框 | D/A 上下张 | S 保存 | U 撤回 | Q 退出')
        print('  当前只补标 person 类')
        print('=' * 50)

    def get_xml_path(self, img_name):
        stem = os.path.splitext(img_name)[0]
        return os.path.join(self.ann_dir, stem + '.xml')

    def load_image(self):
        self.ax.clear()
        self.current_boxes = []
        img_name = self.images[self.idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        self.img_w, self.img_h = img.size
        self.ax.imshow(img)

        # 加载已有标注
        xml_path = self.get_xml_path(img_name)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                bnd = obj.find('bndbox')
                xmin = int(bnd.find('xmin').text)
                ymin = int(bnd.find('ymin').text)
                xmax = int(bnd.find('xmax').text)
                ymax = int(bnd.find('ymax').text)
                color = 'lime' if name == 'person' else 'cyan'
                alpha = 0.6 if name == 'person' else 0.3
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                    linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
                self.ax.add_patch(rect)
                self.ax.text(xmin, ymin-2, name, color=color, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=0))
        else:
            self.ax.set_title('(无XML标注文件)', fontsize=10, color='orange')

        self.ax.set_title(f'[{self.idx+1}/{len(self.images)}] {img_name} ({self.img_w}x{self.img_h}) | 本次已画 {len(self.current_boxes)} 个person框', fontsize=9)
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != MouseButton.LEFT:
            return
        self.start_xy = (int(event.xdata), int(event.ydata))
        self.rect = patches.Rectangle(self.start_xy, 0, 0,
            linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(self.rect)

    def on_motion(self, event):
        if self.rect is None or event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        w = x - self.start_xy[0]
        h = y - self.start_xy[1]
        self.rect.set_width(w)
        self.rect.set_height(h)
        self.rect.set_xy(self.start_xy)
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.rect is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        x1, y1 = self.start_xy
        xmin, xmax = sorted([x1, x])
        ymin, ymax = sorted([y1, y])
        if xmax - xmin < 3 or ymax - ymin < 3:
            self.rect.remove()
        else:
            self.rect.set_bounds(xmin, ymin, xmax-xmin, ymax-ymin)
            self.rect.set_edgecolor('red')
            self.rect.set_linewidth(2)
            self.current_boxes.append((xmin, ymin, xmax, ymax))
            self.ax.set_title(f'[{self.idx+1}/{len(self.images)}] {self.images[self.idx]} | 本次已画 {len(self.current_boxes)} 个person框', fontsize=9)
            self.fig.canvas.draw()
        self.rect = None

    def save(self):
        if not self.current_boxes:
            print('  没有新画的框，跳过保存')
            return
        img_name = self.images[self.idx]
        xml_path = self.get_xml_path(img_name)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
        else:
            root = ET.Element('annotation')
            ET.SubElement(root, 'folder').text = 'images'
            ET.SubElement(root, 'filename').text = img_name
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(self.img_w)
            ET.SubElement(size, 'height').text = str(self.img_h)
            ET.SubElement(size, 'depth').text = '3'
            ET.SubElement(root, 'segmented').text = '0'

        for (xmin, ymin, xmax, ymax) in self.current_boxes:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = 'person'
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'occluded').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bnd = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bnd, 'xmin').text = str(xmin)
            ET.SubElement(bnd, 'ymin').text = str(ymin)
            ET.SubElement(bnd, 'xmax').text = str(xmax)
            ET.SubElement(bnd, 'ymax').text = str(ymax)

        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        print(f'  已保存: {xml_path} (+{len(self.current_boxes)} person框)')

    def on_key(self, event):
        if event.key == 'd':
            self.save()
            self.idx = min(self.idx + 1, len(self.images) - 1)
            self.load_image()
        elif event.key == 'a':
            self.save()
            self.idx = max(self.idx - 1, 0)
            self.load_image()
        elif event.key == 's':
            self.save()
        elif event.key == 'q':
            self.save()
            print('退出')
            plt.close()
            sys.exit(0)


if __name__ == '__main__':
    IMG_DIR = r'D:\program-codex\hw3\hw3\helmet\images'
    CLS_FILE = r'D:\program-codex\hw3\hw3\helmet\classes.txt'
    ANN_DIR = r'D:\program-codex\hw3\hw3\helmet\annotations'
    LabelImgSE(IMG_DIR, CLS_FILE, ANN_DIR)
