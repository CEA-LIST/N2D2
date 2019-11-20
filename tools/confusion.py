#!/usr/bin/python3.7 -u
# -*- coding: ISO-8859-1 -*-
################################################################################
#    (C) Copyright 2016 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

import os, sys
import csv
import numpy

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QHeaderView, QLabel, QHeaderView, QSizePolicy, QTableView, QPushButton, QStyledItemDelegate, QStyle
from PyQt5.QtGui import QIcon, QColor, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QRect, pyqtSlot, pyqtSignal, QAbstractTableModel, QSortFilterProxyModel, QModelIndex

class CustomSortingModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        if left.data() == "" or left.data() is None:
            numLeft = -1
        else:
            numLeft = float(left.data())

        if right.data() == "" or right.data() is None:
            numRight = -1
        else:
            numRight = float(right.data())

        return numLeft > numRight

class CustomTableView(QTableView):
    hoverIndexChanged = pyqtSignal(QModelIndex) 
    leaveTableEvent = pyqtSignal() 

    def __init__(self):
        super(CustomTableView, self).__init__()
        self.myDelegate = MyRowHoverDelegate(self)
        self.setMouseTracking(True)
        self.setStyleSheet("QTableView::item:hover {  border-top: 1px solid; border-bottom: 1px solid; }")

        self.hoverIndexChanged.connect(lambda index: self.myDelegate.onHoverIndexChanged(index))
        self.leaveTableEvent.connect(self.myDelegate.onLeaveTableEvent)
        self.setItemDelegate(self.myDelegate)

    def mouseMoveEvent(self, event):
        index = self.indexAt(event.pos())
        self.hoverIndexChanged.emit(index)
        self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.leaveTableEvent.emit()
        self.viewport().update()
        super().leaveEvent(event)

class MyRowHoverDelegate(QStyledItemDelegate):
    def __init__(self, parent):
        super(MyRowHoverDelegate, self).__init__(parent)
        self.hoverRow = 0

    @pyqtSlot()
    def onHoverIndexChanged(self, index):
        self.hoverRow = index.row()

    @pyqtSlot()
    def onLeaveTableEvent(self):
        self.hoverRow = -1

    def paint(self, painter, option, index):
        opt = option

        #print("%d - %d" % (index.row(), self.hoverRow))
        if index.row() == self.hoverRow:
            opt.state |= QStyle.State_MouseOver
        else:
            opt.state &= ~QStyle.State_MouseOver

        super().paint(painter, opt, index)

class App(QWidget):
    def __init__(self, argv):
        super().__init__()
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 600
        self.transpose = False
        self.percent = False
        self.legend = None

        if len(argv) > 1:
            self.confusionFile = argv[1]
        elif os.path.basename(__file__) != "confusion.py":
            self.confusionFile = os.getcwd()
            self.confusionFile = os.path.join(self.confusionFile,
                os.path.splitext(os.path.basename(__file__))[0] + ".dat")

        self.title = "Confusion Matrix [%s]" % (self.confusionFile)

        if len(argv) > 2:
            self.legendFile = argv[2]
        elif os.path.basename(__file__) != "confusion.py" and \
          os.path.isfile("labels_mapping.log.dat"):
            self.legendFile = "labels_mapping.log.dat"
        else:
            self.legendFile = None

        self.initUI(self.confusionFile, self.legendFile)

    def initUI(self, confusionFile, legendFile=None):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.createTable(confusionFile, legendFile)

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QHBoxLayout()
        self.topLayout = QVBoxLayout()
        self.topVLayout = QHBoxLayout()
        self.layout = QHBoxLayout()
        self.transposeButton = QPushButton("M\u1D40")
        self.transposeButton.setFixedWidth(25)
        self.transposeButton.setCheckable(True)
        self.transposeButton.clicked.connect(self.onTranspose)
        self.topVLayout.addWidget(self.transposeButton)
        self.percentButton = QPushButton("%")
        self.percentButton.setFixedWidth(25)
        self.percentButton.setCheckable(True)
        self.percentButton.clicked.connect(self.onPercent)
        self.topVLayout.addWidget(self.percentButton)
        self.permutButton = QPushButton("\u2198")
        self.permutButton.setFixedWidth(25)
        self.permutButton.clicked.connect(self.onPermut)
        self.topVLayout.addWidget(self.permutButton)
        self.restoreButton = QPushButton("\u21BA")
        self.restoreButton.setFixedWidth(25)
        self.restoreButton.clicked.connect(self.onRestore)
        self.topVLayout.addWidget(self.restoreButton)
        self.topLabel = QLabel("Estimated")
        self.topLabel.setStyleSheet("font-weight: bold")
        self.topLabel.setAlignment(Qt.AlignCenter);
        self.topVLayout.addWidget(self.topLabel) 
        self.topLayout.addLayout(self.topVLayout)
        self.tableLayout = QHBoxLayout()
        self.leftLabel = QLabel("T\na\nr\ng\ne\nt")
        self.leftLabel.setStyleSheet("font-weight: bold")
        self.tableLayout.addWidget(self.leftLabel) 
        self.tableLayout.addWidget(self.tableView)
        self.topLayout.addLayout(self.tableLayout)
        self.layout.addLayout(self.topLayout)

        self.sideLayout = QVBoxLayout()
        self.sideLayout.setAlignment(Qt.AlignTop)
        self.targetLabel = QLabel()
        self.targetLabel.setText("Target:")
        self.targetLabel.setFixedWidth(200)
        self.sideLayout.addWidget(self.targetLabel)
        self.estimatedLabel = QLabel()
        self.estimatedLabel.setText("Estimated:")
        self.estimatedLabel.setFixedWidth(200)
        self.sideLayout.addWidget(self.estimatedLabel)

        self.confTable = QTableWidget()
        self.confTable.setFixedWidth(200)
        self.confTable.setFixedHeight(120)
        self.confTable.setRowCount(2)
        self.confTable.setColumnCount(2)
        self.confTable.setVerticalHeaderLabels(["cls", "non\n-cls"])
        self.confTable.setHorizontalHeaderLabels(["Est. cls", "Est. non-cls"])
        self.confTable.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.confTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.confTable.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)
        self.sideLayout.addWidget(self.confTable)

        self.score = QLabel()
        self.score.setFixedWidth(200)
        self.sideLayout.addWidget(self.score)

        self.sideLayout.addStretch() # Adding a stretch to occupy the empty space
        self.layout.addLayout(self.sideLayout)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def loadLegend(self, fileName):
        with open(fileName, 'r') as f:
            legend = csv.reader(f, delimiter=' ', doublequote=False, escapechar='\\',
                                 strict=True)
            next(legend, None)  # skip the headers
            legend = list(zip(*legend))
            self.legend = dict(zip(list(map(int, legend[2])), legend[1]))

    def loadConfusion(self, fileName):
        with open(fileName, 'r') as f:
            conf = csv.reader(f, delimiter=' ', doublequote=False, escapechar='\\',
                                 strict=True)
            next(conf, None)  # skip the headers
            conf = list(conf)
            n = int(numpy.sqrt(len(conf)))

            self.confArray = numpy.zeros([n,n])
            for x in conf:
                self.confArray[int(x[0]), int(x[1])] = int(x[2])

    def createTable(self, confusionFile, legendFile=None):
        # Create table
        self.loadConfusion(confusionFile)

        if legendFile is not None:
            self.loadLegend(legendFile)

        font = QFont()
        font.setPointSize(7)

        self.tableModel = QStandardItemModel()
        self.proxyModel = CustomSortingModel()
        self.proxyModel.setSourceModel(self.tableModel)

        n = len(self.confArray[0])
        self.tableView = QTableView()
        self.tableView.setModel(self.proxyModel)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.horizontalHeader().setFont(font)
        self.tableView.verticalHeader().setFont(font)
        self.tableModel.setRowCount(n + 1)
        self.tableModel.setColumnCount(n + 3)
        self.tableView.horizontalHeader().setSectionResizeMode(n, QHeaderView.ResizeToContents)
        self.tableView.horizontalHeader().setSectionResizeMode(n + 1, QHeaderView.ResizeToContents)
        self.tableView.horizontalHeader().setSectionResizeMode(n + 2, QHeaderView.ResizeToContents)
        self.tableView.setSortingEnabled(True)

        self.fillTableLabels(self.legend)
        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)

        self.populateTable(n)
        self.fillTable(self.confArray)

        # table selection change
        self.tableView.selectionModel().selectionChanged.connect(self.onSelectionChanged)

    def populateTable(self, n):
        font = QFont()
        font.setPointSize(7)

        for target in range(0, n):
            for estimated in range(0, n):
                item = QStandardItem("")
                item.setFont(font)
                self.tableModel.setItem(target, estimated, item)

            item = QStandardItem("")
            item.setFont(font)
            self.tableModel.setItem(target, n, item)
            item = QStandardItem("")
            item.setFont(font)
            self.tableModel.setItem(target, n + 1, item)
            item = QStandardItem("")
            item.setFont(font)
            self.tableModel.setItem(target, n + 2, item)
            item = QStandardItem("")
            item.setFont(font)
            self.tableModel.setItem(n, target, item)

        item = QStandardItem("")
        item.setFont(font)
        self.tableModel.setItem(n, n, item)

    def fillTableLabels(self, legend):
        n = self.tableModel.rowCount() - 1

        if legend is not None:
            labels = ["%d: %s" % (k, legend[k]) for k in range(0, n)]
        else:
            labels = ["%d" % (k) for k in range(0, n)]

        labels.append("Count")
        self.tableModel.setVerticalHeaderLabels(labels)
        labels.append("Recall")
        labels.append("Precision")
        labels = ['\n'.join(label[i:i + 4] for i in range(0, len(label), 4)) for label in labels]
        self.tableModel.setHorizontalHeaderLabels(labels)

    def fillTable(self, confArray, transpose=False, percent=False):
        total = 0

        colCount = []
        for col in confArray.T:
            count = numpy.sum(col)
            colCount.append(count)
            total += count

        for target, row in enumerate(confArray):
            count = numpy.sum(row)
            tp = row[target]
            fn = count - tp
            recall = tp / float(tp + fn) if count > 0 else 0.0

            for estimated in range(0, len(row)):
                if transpose:
                    item = self.tableModel.item(estimated, target)
                    fracCol = confArray[target, estimated] / float(count)
                    fracRow = confArray[target, estimated] / float(colCount[estimated])
                else:
                    item = self.tableModel.item(target, estimated)
                    fracCol = confArray[target, estimated] / float(colCount[estimated])
                    fracRow = confArray[target, estimated] / float(count)

                item.setText("")

                if percent:
                    if fracCol > 1.0e-3:
                        item.setText("%.03f" % (fracCol))
                elif int(confArray[target, estimated]) > 0:
                    item.setText(str(int(confArray[target, estimated])))

                c1 = 255 * (1.0 - fracRow)
                if target == estimated:
                    c2 = 255 * fracRow
                    item.setBackground(QColor(c1, c2, c2))
                else:
                    item.setBackground(QColor(255, c1, c1))

            item = self.tableModel.item(target, len(row) + 1)
            item.setText("%.03f" % (recall))

            if recall > 0.5:
                c = 255 * (1.0 - 2.0 * (recall - 0.5))
                item.setBackground(QColor(c, 255, c))
            else:
                c = 255 * (1.0 - 2.0 * (0.5 - recall))
                item.setBackground(QColor(255, c, c))

            if self.transpose:
                countItem = self.tableModel.item(len(row), target)
            else:
                countItem = self.tableModel.item(target, len(row))

            if percent:
                countItem.setText("%.03f" % (count / float(total)))
            else:
                countItem.setText("%d" % (count))

            countItem.setBackground(QColor(255, 255, 0))

        totalItem = self.tableModel.item(len(confArray[0]), len(confArray[0]))
        totalItem.setText("%d" % (total))
        totalItem.setBackground(QColor(255, 165, 0))

        for target, col in enumerate(confArray.T):
            count = colCount[target]
            tp = col[target]
            fp = count - tp
            precision = tp / float(tp + fp) if count > 0 else 0.0

            item = self.tableModel.item(target, len(col) + 2)
            item.setText("%.03f" % (precision))

            if precision > 0.5:
                c = 255 * (1.0 - 2.0 * (precision - 0.5))
                item.setBackground(QColor(c, 255, c))
            else:
                c = 255 * (1.0 - 2.0 * (0.5 - precision))
                item.setBackground(QColor(255, c, c))

            if self.transpose:
                countItem = self.tableModel.item(target, len(col))
            else:
                countItem = self.tableModel.item(len(col), target)

            countItem.setText("%d" % (count))
            countItem.setBackground(QColor(255, 255, 0))

        self.tableView.resizeRowsToContents()


    @pyqtSlot()
    def onSelectionChanged(self):
        total = numpy.sum(self.confArray)
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        selTargets = []
        selEstimated = []

        for currentQTableWidgetItem in self.tableView.selectedIndexes():
            if self.transpose:
                target = self.tableView.model().mapToSource(currentQTableWidgetItem).column()
            else:                
                target = self.tableView.model().mapToSource(currentQTableWidgetItem).row()

            if not target in selTargets and target in self.legend:
                selTargets.append(target)

            if self.transpose:
                estimated = self.tableView.model().mapToSource(currentQTableWidgetItem).row()
            else:                
                estimated = self.tableView.model().mapToSource(currentQTableWidgetItem).column()

            if not estimated in selEstimated and estimated in self.legend:
                selEstimated.append(estimated)

        if len(selTargets) == 1:
            self.targetLabel.setText("Target: %d (%s)" % (selTargets[0], self.legend[selTargets[0]]))
        else:
            self.targetLabel.setText("Target:")

        if len(selEstimated) == 1:
            self.estimatedLabel.setText("Estimated: %d (%s)" % (selEstimated[0], self.legend[selEstimated[0]]))
        else:
            self.estimatedLabel.setText("Estimated:")

        if len(selTargets) > 0:
            for t1 in selTargets:
                for t2 in selTargets:
                    tp += self.confArray[t1, t2]

            for target in selTargets:
                fp += numpy.sum(self.confArray[:, target])
            fp -= tp

            for target in selTargets:
                fn += numpy.sum(self.confArray[target,:])
            fn -= tp

            tn = total - tp - fp - fn
            recall = tp / float(tp + fn)
            precision = tp / float(tp + fp)

            tpItem = QTableWidgetItem("%d\n(true pos.)" % (tp))
            c1 = 255 * (1.0 - recall)
            c2 = 255 * recall
            tpItem.setBackground(QColor(c1, c2, c2))

            c3 = 255 * precision
            fpItem = QTableWidgetItem("%d\n(false pos.)" % (fp))
            fpItem.setBackground(QColor(255, c3, c3))

            fnItem = QTableWidgetItem("%d\n(false neg.)" % (fn))
            fnItem.setBackground(QColor(255, c2, c2))

            tnItem = QTableWidgetItem("%d\n(true neg.)" % (tn))

            self.confTable.setItem(0, 0, tpItem)
            self.confTable.setItem(1, 0, fpItem)
            self.confTable.setItem(0, 1, fnItem)
            self.confTable.setItem(1, 1, tnItem)

            cls = "\n     ".join(self.legend[t] for t in selTargets)
            self.score.setText("cls: %s\nRecall: %f\nPrecision: %f" % (cls, recall, precision))
        else:
            self.confTable.setItem(0, 0, QTableWidgetItem(""))
            self.confTable.setItem(1, 0, QTableWidgetItem(""))
            self.confTable.setItem(0, 1, QTableWidgetItem(""))
            self.confTable.setItem(1, 1, QTableWidgetItem(""))
            self.score.setText("")

    def onTranspose(self):
        self.transpose = not self.transpose

        if self.transpose:
            self.topLabel.setText("Target")
            self.leftLabel.setText("E\ns\nt\ni\nm\na\nt\ne\nd")
        else:
            self.topLabel.setText("Estimated")
            self.leftLabel.setText("T\na\nr\ng\ne\nt")

        self.fillTable(self.confArray, transpose=self.transpose, percent=self.percent)

    def onPercent(self):
        self.percent = not self.percent
        self.fillTable(self.confArray, transpose=self.transpose, percent=self.percent)

    def onRestore(self):
        self.loadConfusion(self.confusionFile)

        self.legend = None
        if self.legendFile is not None:
            self.loadLegend(self.legendFile)

        self.fillTableLabels(self.legend)
        self.fillTable(self.confArray, transpose=self.transpose, percent=self.percent)

        self.tableView.horizontalHeader().setSortIndicator(-1, Qt.DescendingOrder)
        self.proxyModel.sort(-1)

    def onPermut(self):
        srcRows = [x for x in range(0, self.tableModel.rowCount() - 1)]
        destRows = [None] * (self.tableModel.rowCount() - 1)

        for row in srcRows:
            index = self.tableView.model().index(row, 0)
            modelIndex = self.tableView.model().mapToSource(index)
            destRows[modelIndex.row()] = row

        if self.legend is not None:
            legend = [[destRows[k], v] for k, v in self.legend.items()]
        else:
            legend = [[destRows[k], k] for k in range(0, self.tableModel.rowCount() - 1)]

        self.legend = dict(legend)

        self.confArray[destRows,:] = self.confArray[srcRows,:]
        self.confArray[:,destRows] = self.confArray[:,srcRows]

        self.fillTableLabels(self.legend)
        self.fillTable(self.confArray, transpose=self.transpose, percent=self.percent)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App(sys.argv)
    sys.exit(app.exec_())  

