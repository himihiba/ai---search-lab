import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.nn.metrics import Accuracy
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
DATA_DIR_TRAIN = r"/media/sf_forward_mnist/MNIST/train"
DATA_DIR_TEST = r"/media/sf_forward_mnist/MNIST/test"

ds_train = ds.MnistDataset(os.path.join(DATA_DIR_TRAIN))
ds_test = ds.MnistDataset(os.path.join(DATA_DIR_TEST))
print('number of training datasets: ', ds_train.get_dataset_size())
print('number of test datasets: ', ds_test.get_dataset_size())
image=ds_train.create_dict_iterator().__next__()
print('image length/width/number of channels: ', image['image'].shape)
print(' label of an image: ',image['label'])
def create_dataset(training=True, batch_size=128, resize=(28,28), rescale=1/255, shift=0, buffer_size=64):
    ds = ms.dataset.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)
    resize_op = CV.Resize(resize)
    rescale_op = CV.Rescale(rescale,shift)
    hwc2chw_op = CV.HWC2CHW()
    ds = ds.map(input_columns="image", operations=[rescale_op,resize_op, hwc2chw_op])
    ds = ds.map(input_columns="label", operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
ds = create_dataset(training=False)
data = ds.create_dict_iterator().__next__()
images = data['image'].asnumpy()
labels = data['label'].asnumpy()
for i in range(1,11):
    plt.subplot(2, 5, i)
    plt.imshow(np.squeeze(images[i]))
    plt.title('Number: %s' % labels[i])
    plt.xticks([])
plt.savefig("/media/sf_code/mnist_preview.png")
class ForwardNN(nn.Cell):
    def __init__(self):
        super(ForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(784, 512, activation='relu')
        self.fc2 = nn.Dense(512, 128, activation='relu')
        self.fc3 = nn.Dense(128, 10, activation='softmax')
    def construct(self, input_x):
        output = self.flatten(input_x)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
lr = 0.001
num_epoch = 10
momentum = 0.9
net = ForwardNN()
loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
metrics={"Accuracy": Accuracy()}
opt = nn.Adam(net.trainable_params(), lr)
model = Model(net, loss, opt, metrics)
config_ck = CheckpointConfig(save_checkpoint_steps=1875,keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_net",directory="./ckpt" ,config=config_ck)
ds_eval = create_dataset(False, batch_size=32)
ds_train = create_dataset(batch_size=32)
loss_cb = LossMonitor(per_print_times=1875)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
print("==========starting Training==========")
model.train(num_epoch, ds_train,callbacks=[ckpoint_cb,loss_cb,time_cb],dataset_sink_mode=False)

metrics=model.eval(ds_eval)
print(metrics)





# *********************** #

#Create a drawing window for writing your own handwritten digits



import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
class PenWidthDlg(QDialog):
    def __init__(self, parent=None):
        super(PenWidthDlg, self).__init__(parent)

        widthLabel = QLabel("Width:")
        self.widthSpinBox = QSpinBox()
        widthLabel.setBuddy(self.widthSpinBox)
        self.widthSpinBox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.widthSpinBox.setRange(0, 500)

        okButton = QPushButton("ok")
        cancelButton = QPushButton("cancel")

        layout = QGridLayout()
        layout.addWidget(widthLabel, 0, 0)
        layout.addWidget(self.widthSpinBox, 0, 1)
        layout.addWidget(okButton, 1, 0)
        layout.addWidget(cancelButton, 1, 1)
        self.setLayout(layout)
        self.setWindowTitle("Width Setting")

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)


class myMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("draw")
        self.pix = QPixmap()
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.initData()
        self.initView()

        self.Menu = self.menuBar().addMenu("Menu")

        self.ClearAction = QAction(QIcon("images/clear.png"), "Clear", self)
        self.ClearAction.triggered.connect(self.initView)
        self.Menu.addAction(self.ClearAction)

        self.changeWidth = QAction(QIcon("images/width.png"), "Width", self)
        self.changeWidth.triggered.connect(self.showWidthDialog)
        self.Menu.addAction(self.changeWidth)

        self.fileSaveAction = QAction(QIcon("images/filesave.png"), "&Save", self)
        self.fileSaveAction.setShortcut(QKeySequence.Save)
        self.fileSaveAction.triggered.connect(self.fileSaveAs)
        self.Menu.addAction(self.fileSaveAction)

    def initData(self):
        self.size = QSize(1120, 1120)
        self.pixmap = QPixmap(self.size)

        self.dirty = False
        self.filename = None
        self.recentFiles = []

        
        self.width = 100
        self.color = QColor("black")
        self.pen = QPen()      
        self.pen.setColor(self.color)  
        self.pen = QPen(Qt.SolidLine)  
        self.pen.setWidth(self.width)  

        
        self.painter = QPainter(self.pixmap)
        self.painter.setPen(self.pen)

        
        self._lastPos = QPoint(0, 0)    
        self._currentPos = QPoint(0, 0) 
        self.image = QImage()



    def initView(self):
        self.Clear()
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(self.pixmap)
        self.setCentralWidget(self.imageLabel)



    def Clear(self):
        self.pixmap.fill(Qt.white)
        self.update()
        self.dirty = False



    def mousePressEvent(self, event):

        pointX = event.globalX()
        pointY = event.globalY()
        self._currentPos = QPoint(pointX, pointY)
        self.dirty = True
        self._currentPos = event.pos()
        self._lastPos = self._currentPos



    def mouseMoveEvent(self, event):
    
        self._currentPos = event.pos()

    
        self.painter.begin(self.pixmap)
        self.painter.setPen(self.pen)
        self.painter.drawLine(self._lastPos, self._currentPos)

        self._lastPos = self._currentPos
        self.painter.end()
        self.update()   
        self.imageLabel.setPixmap(self.pixmap)



    def updateWidth(self):
        self.pen.setWidth(self.width)
        self.painter.setPen(self.pen)



    def showWidthDialog(self):
        dialog = PenWidthDlg(self)
        dialog.widthSpinBox.setValue(self.width)
        if dialog.exec_():
            self.width = dialog.widthSpinBox.value()
            self.updateWidth()



    def fileSaveAs(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.', "*.png")
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.pixmap
        print("save...")
        image = image.scaled(28, 28)
        image.save(savePath[0])


app = QApplication(sys.argv)
form = myMainWindow()
form.setMinimumSize(1120, 1120)
form.show()
app.exec_()

from PIL import Image, ImageOps


image = Image.open('/media/sf_code/images/4.png')


image = image.convert('L')
in_image = ImageOps.invert(image)




plt.figure('image')
plt.imshow(in_image)


new_image = np.array(in_image)
new_image = new_image / 255
new_image = new_image.reshape(1, 1, 28, 28)


input_image = ms.Tensor(new_image, dtype=ms.float32)
result = model.predict(input_image)


print(result)
result = ms.ops.Argmax()(result)
print(result)

