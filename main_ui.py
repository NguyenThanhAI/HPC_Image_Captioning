# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, img, word_map, rev_word_map, beam_size=3):

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    #img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    words = [rev_word_map[ind] for ind in seq]


    return words

class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self, args) -> None:
        super().__init__()
        self.loaded_image = False
        self.loaded_model = False
        self._translate = QtCore.QCoreApplication.translate

        self.image = None

        self.args = args

    
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 650, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.run_model)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 650, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.load_model)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 800, 600))
        self.label.setObjectName("label")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(200, 700, 120, 20))
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 750, 120, 20))
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(850, 10, 300, 200))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setReadOnly(True)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(200, 650, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.load_image)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(850, 250, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.clear_image)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Nhóm 6 - Sinh mô tả tự động cho ảnh"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.pushButton_2.setText(_translate("MainWindow", "Load model"))
        #self.label.setText(_translate("MainWindow", "Image"))
        self.label.setStyleSheet("border: 1px solid black")
        #self.label_1.setText(_translate("MainWindow", "No Image Loaded"))
        #self.label_1.setStyleSheet("background-color: red; border: 1px solid black")
        #self.label_2.setText(_translate("MainWindow", "Model not loaded"))
        #self.label_2.setStyleSheet("background-color: red; border: 1px solid black")
        self.off_image()
        self.off_model()
        self.pushButton_3.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_4.setText(_translate("MainWindow", "Clear"))

    def off_image(self):
        self.label_1.setText(self._translate("MainWindow", "No Image Loaded"))
        self.label_1.setStyleSheet("background-color: red; border: 1px solid black")

    def on_image(self):
        self.label_1.setText(self._translate("MainWindow", "Image Loaded"))
        self.label_1.setStyleSheet("background-color: lightgreen; border: 1px solid black")

    def off_model(self):
        self.label_2.setText(self._translate("MainWindow", "Model not loaded"))
        self.label_2.setStyleSheet("background-color: red; border: 1px solid black")

    def on_model(self):
        self.label_2.setText(self._translate("MainWindow", "Model loaded"))
        self.label_2.setStyleSheet("background-color: lightgreen; border: 1px solid black")

    
    def load_image(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "Image files (*.jpg)")
        imagePath = fname[0]
        self.image = cv2.imread(imagePath)
        try:
            self.image = self.image[:, :, ::-1]
        except TypeError:
            return
        pixmap = QtGui.QPixmap(imagePath)

        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        #self.label.setSizePolicy(QtWidgets.QSizePolicy)
        #self.resize(pixmap.width(), pixmap.height())
        self.loaded_image = True
        #self.label_1.setText(self._translate("MainWindow", "Image Loaded"))
        #self.label_1.setStyleSheet("background-color: lightgreen; border: 1px solid black")
        self.on_image()

    def load_model(self):
        if not self.loaded_model:
            self.loaded_model = True
            #self.label_2.setText(self._translate("MainWindow", "Model loaded"))
            #self.label_2.setStyleSheet("background-color: lightgreen; border: 1px solid black")
            with open(self.args.word_map, 'r') as j:
                self.word_map = json.load(j)
            self.rev_word_map = {v: k for k, v in self.word_map.items()}

            checkpoint = torch.load(self.args.model, map_location=torch.device("cpu"))
            #encoder = Encoder()
            #encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder = checkpoint["encoder"]
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
            #decoder = Decoder(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=len(rev_word_map) + 2)
            #decoder.load_state_dict(checkpoint["decoder_state_dict"])
            self.decoder = checkpoint["decoder"]
            self.decoder = self.decoder.to(device)
            self.decoder.eval()
            self.on_model()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Model loaded before")
            msg.setWindowTitle("Model loaded")
            msg.exec_()

    def run_model(self):
        if self.loaded_image and self.loaded_model:
            caption = caption_image_beam_search(self.encoder, self.decoder, self.image, self.word_map, self.rev_word_map, beam_size=self.args.beam_size)
            caption = " ".join(caption[1:-1])
            self.lineEdit.setText(caption)
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("You must load image and model to run model")
            if self.loaded_image and not self.loaded_model:
                msg.setInformativeText("Load image but not load model yet")
            elif not self.loaded_image and self.loaded_model:
                msg.setInformativeText("Load model but not load image yet")
            else:
                msg.setInformativeText("Both not load image and model yet")
            msg.setWindowTitle("Error")
            msg.exec_()

    def clear_image(self):
        if self.loaded_image:
            self.label.clear()
            self.image = None
            #_translate = QtCore.QCoreApplication.translate
            #self.label.setText(self._translate("MainWindow", "Image"))
            self.loaded_image = False
            #self.label_1.setText(self._translate("MainWindow", "No Image Loaded"))
            #self.label_1.setStyleSheet("background-color: red; border: 1px solid black")
            self.off_image()

        self.lineEdit.clear()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', help='path to model', default=r"C:\Users\Thanh\Downloads\Image_Captioning_Checkpoint\BEST_checkpoint_caption_model.pth.tar")
    parser.add_argument('--word_map', '-wm', help='path to word map JSON', default=r"C:\Users\Thanh\Downloads\Image_Captioning_Checkpoint\tokenizer.json")
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')

    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(args)
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())
