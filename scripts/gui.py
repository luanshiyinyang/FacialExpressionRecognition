import sys
from PyQt5 import QtWidgets

sys.path.append('ui')
from ui import UI


def load_cnn_model():
    """
    载入CNN模型
    :return:
    """
    from model import CNN3
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
