"""
@author: lataf 
@file: health_sys.py
@time: 12.04.2024 13:49
Модуль отвечает за консолидацию модулей
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""
import abc
import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_application_parameters():
    # pd.options.mode.chained_assignment = None  # Убирает предупреждение о запутанности
    plt.rcParams["figure.figsize"] = (9, 9)  # Размер графика в дюймах ( ширина, длина)
    plt.rcParams['axes.grid'] = True  # Наличие сетки
    plt.rcParams['figure.facecolor'] = 'white'  # Белый фон графика
    plt.rcParams['lines.linestyle'] = '-'  # Установить стиль линии
    plt.rcParams['lines.linewidth'] = 3  # Установить ширину линии
    pd.set_option('display.max_columns', 20)  # Число показываемых колонок
    pd.set_option('display.width', 200)  # Ширина выводимой таблицы
    pd.set_option('display.float_format', '{:.0f}'.format)  # Показ целых в Датафрейме


set_application_parameters()


def str_main():
    import streamlit as st
    # from streamlit.web.cli import main
    # import sys

    # import numpy as np
    # import matplotlib.pyplot as plt

    st.title("Калькулятор здоровья")
    page = st.sidebar.selectbox("Подсистема организма",
                                ["Жировой запас",
                                 "Сердце",
                                 "Легкие"
                                 ])
    st.write(f'Ваш ИМТ = 22')

    if page == "Жировой запас":
        st.header("""Индекс массы тела (ИМТ)""")
        st.text("Для расчета индекса массы тела введите свой:")
        weight = st.number_input(' вес в килограммах', value=72, placeholder="Вес в килограммах")
        height = st.number_input(' рост в сантиметрах', value=170, placeholder="Рост в см")
        # number = st.number_input("Insert a number", value=None, placeholder="Type a number...")

        if st.button('Рассчитать ИМТ'):
            _imt = weight / ((height / 100) ** 2)
            _imt = round(_imt, 2)
            st.write(f'Ваш ИМТ = {_imt}')

    elif page == "Сердце":
        st.header("""Сердце:""")
    elif page == "Легкие":
        st.header("""Легкие:""")

        # st.latex(r'''
        #             F(x) = exp(-(\gamma x)^{-1/\gamma}1\{x>0\})
        #             ''')
        # st.text("Для получения результата:")
        # st.markdown(
        #     "* Сгенерируем N нормально распределенных случайных величин $U_i$ [0,1] (среднее и единичная дисперсия).")
        # st.markdown("* Вычислим N  величин с распределением по формуле:")
        # st.latex(r'''
        #                     X_i=\dfrac{1}{\gamma}\left(-lnU_i)^{-\gamma}\right)
        #                 ''')
        # mu, sigma = 0, 1  # mean and standard deviation
        # gamma = st.slider('Желаемая гамма', 0.25, 2.25, 0.5, 0.25)
        # N = st.number_input("Желаемое N", 100, 10000, 10000)
        # U = np.abs(np.random.normal(mu, sigma, N))
        # X = 1 / gamma * (-np.log(U)) ** (-gamma)
        # X2 = X[X < 20]
        # fig, ax = plt.subplots()
        # count, bins, ignored = plt.hist(X2, 100, density=True)
        # plt.plot(bins,
        #          np.exp(- (gamma * bins) ** (-1 / gamma)) * (1 / gamma) * (gamma * bins) ** (-1 / gamma - 1) * gamma,
        #          linewidth=2, color='r')
        # st.pyplot(fig)


view_console = True


class User:
    """Вводи и вывод данных пользователем """

    def __init__(self):
        self.age = 30
        self.gender = 'women'
        self.health: Health = Health(self)  # Система здоровья

    def input(self):
        ...

    @staticmethod
    def output(val):
        plt.show()
        print(f'Сердце - {val}')


class Subsys(abc.ABC):
    def __init__(self):
        self.harrington = Harrington()
        self.health = None  # Ссылка на родителя
        self.data = ''  # Название файла json с загрузочными данными
        self.name = ''  # Название параметра
        self.current_value = None  # Текущее показание
        self.h_level = None  # Показатель Харрингтона

    @abc.abstractmethod
    def load(self, json_name):
        """Загружаем начальные данные"""
        ...

    @abc.abstractmethod
    def calc(self):
        """Считаем функцию желательности"""
        ...

    def calibrate(self, json_name: str, x: int, z: int) -> plt:
        """Калибровочная диаграмма"""
        self.harrington.data(json_name)
        self.harrington.load()
        with open(json_name, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        imt_range = range(self.data["range"]["begin"], self.data["range"]["end"], 1)

        d_range_1 = []
        for y in imt_range:
            d = self.harrington.calc(y)
            d_range_1.append(d * 100)
        plt.plot(imt_range, d_range_1, label="Калибровка", marker="o", ms=6, mfc='w')
        # plt.grid()
        plt.title(f'Калибровочная диаграмма "{self.data["name"]}"')
        plt.ylabel(f'Желательность параметра ", %', loc='top', fontsize=12)
        plt.xlabel(f'Значение параметра ', loc='right', fontsize=12)
        plt.axhline(y=20, color='black', linestyle='--')
        plt.text(self.data["range"]["begin"], 15, 'Плохо', fontsize=15)
        plt.axhline(y=80, color='black', linestyle='--')
        plt.text(self.data["range"]["begin"], 75, 'Хорошо', fontsize=15)
        plt.legend(loc='best')
        plt.plot(x, z, 'ro', markersize=12, )
        plt.text(x + 4, z - 2, '  Ваше\nзначение', fontsize=15)
        if view_console:
            plt.show()  # Показываем график в приложении
        return plt


class Health:
    """Управление объектами """

    def __init__(self, _user: User = None):
        self.user = _user  # Ссылка на родителя
        # self.pulse = Pulse(self)  # ресурсы сердечно-сосудистой системы
        # self.imt = IMT(self)  # индекс массы тела
        # self.resp = Resp(self)  # ресурсы легких
        # self.harrington = Harrington1(self)  # Односторонний перевод параметра в безразмерную величину
        # self.harrington2 = Harrington2(self)  # перевод параметра в безразмерную величину
        # self.harrington = HarringtonOne(self)  # Односторонний перевод параметра в безразмерную величину
        # self.harrington2 = HarringtonTwoOne(self)  # перевод параметра в безразмерную величину
        self.subsystems: dict[str, Subsys] = dict()

    def add_subsystem(self, subsystem: Subsys):
        # json_file = subsystem.__class__.__name__.lower() + '.json'
        # subsystem.health = self
        # subsystem.load(json_file)
        # self.subsystems[subsystem.__class__.__name__.lower()] = subsystem
        self.subsystems[subsystem.name] = subsystem

    @staticmethod
    def create_diagram(params: list, results: list):
        """Показываем диаграмму"""
        # params = names  # Названия параметров
        # results = results  # Значения параметров

        theta = np.linspace(start=0, stop=2 * np.pi, num=len(results), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        results = np.append(results, results[0])
        fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]},
                                     figsize=(10, 5), facecolor='#f3f3f3')
        a0.axes.get_xaxis().set_visible(False)  # Убираем надписи на осях
        a0.axes.get_yaxis().set_visible(False)
        a1.axes.get_xaxis().set_visible(False)
        a1.axes.get_yaxis().set_visible(False)
        ax = fig.add_subplot(121, projection='polar')
        ax.plot(theta, results, linewidth=4, color="red", marker='o')
        ax.set_thetagrids(range(0, 360, int(360 / len(params))), params)
        plt.yticks(np.arange(0, 110, 10), fontsize=10)
        ax.set(facecolor='#f3f3f3')
        ax.set_theta_offset(np.pi / 2)
        pl = ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        text_block = Health.text_block(params, results)

        plt.subplot(1, 2, 2)
        plt.text(0.05, 0.5, text_block, fontsize=14)  # Выводим сводный показатель и отдельные показатели
        if view_console:
            plt.show()  # Показываем график в приложении
        return plt

    @staticmethod
    def text_block(par, res) -> str:
        """Создаем текстовой блок в диаграмме"""
        health = int(round(math.prod(res) ** (1 / len(res)), 0))  # Обобщенный показатель Харрингтона
        header = f"На {datetime.today().strftime('%Y-%m-%d')}\nобщее здоровье {health}% \n "  # Заголовок
        parameters = ''  # Показатели здоровья
        for i in range(len(res) - 1):
            parameters += f'    {i + 1}.  {par[i]} {res[i]}%\n'  # Собираем все показатели в строку
        text_block = f'{header}{parameters}'
        return text_block


class Harrington:
    """Односторонний критерий Харрингтона """

    def __init__(self, _subsys: Subsys = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        self.health = _subsys  # Ссылка на родителя
        self.type = 'one'  # one, max, min
        self.h_level = None

    def data(self, json_name: str):
        ...

    def load(self):
        ...

    def calc(self, y):
        return int(y)


class HarringtonParabola(Harrington):
    """Односторонний критерий Харрингтона """

    def __init__(self, _subsys: Subsys = None, x_good=1, x_bad=0):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        # self.type = 'one'  # one, max, min
        self.y_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
        self.y_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
        self.h_good = -math.log(math.log(1 / self.y_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / self.y_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.x_good = x_good  # Назначаем "хороший" параметр Y при self.d_good
        self.x_bad = x_bad  # Назначаем "плохой" параметр Y при self.d_bad
        self.x_optimum = 80
        self.y_optimum = 1
        self.load()  # Считаем коэффициенты в уравнении Харрингтона
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y
        self.a = 1
        self.x_level = 1

    def load(self):
        # Парабола y = ax**2 + bx + c или
        # y = a(x + q)**2 + z, где  q = b/2a,  z = - (b**2 - 4ac) / 4a

        # Ахназарова с. 207   h_level = exp [—ехр(— у')], где у’ = bo + b1 * у'
        self.b_1 = (self.h_good - self.h_bad) / (self.x_good - self.x_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * self.x_good  # Считаем b_0 из уравнений (1) и (2)
        self.a = self.y_optimum / self.x_optimum ** 2
        self.x_level = self.h_bad + (self.x_good - self.h_bad) / 1.25
        # self.x_level = self.h_good + (self.x_bad - self.h_good) / 1.25

    def calc(self, x: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        if x < self.x_level:
            self.h_level = math.exp(-math.exp(-(self.b_0 + self.b_1 * x)))  # Считаем d по Ахназаровой с.207
        else:
            self.h_level = -self.a * (x - self.x_optimum) ** 2 + self.y_optimum  #
        return self.h_level  # Частная функция желательности d


class HarringtonOne(Harrington):
    """Односторонний критерий Харрингтона """

    def __init__(self, _subsys: Subsys = None, y_good=1, y_bad=0):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        # self.type = 'one'  # one, max, min
        self.d_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
        self.d_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
        self.h_good = -math.log(math.log(1 / self.d_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / self.d_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.y_good = y_good  # Назначаем "хороший" параметр Y при self.d_good
        self.y_bad = y_bad  # Назначаем "плохой" параметр Y при self.d_bad
        self.load()  # Считаем коэффициенты в уравнении Харрингтона
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y

    def load(self):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.b_1 = (self.h_good - self.h_bad) / (self.y_good - self.y_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * self.y_good  # Считаем b_0 из уравнений (1) и (2)

    def calc(self, y: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.h_level = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
        return self.h_level  # Частная функция желательности d


class HarringtonHalfTwo(Harrington):
    """Половина двухстороннего критерия Харрингтона"""

    def __init__(self, _subsys: Subsys = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        # self.health = _health  # Ссылка на родителя
        self.d_good = 0.8  # Хороший результат по Харрингтону
        self.y_good = 25  # 65
        self.d_bad = 0.2  # Хороший результат по Харрингтону
        self.y_bad = 45  # 10
        self.opt_d = 1
        self.y_optimum = 20.5
        self.d: float = 0  # Частная функция желательности Харрингтона для параметра y
        self.d_param = 0.75  # 0.6
        self.y1 = None
        self.y = None
        self.n = None
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y

    def data(self, json_name):
        with open(json_name, 'r') as f:
            data = json.load(f)
        # self.y_good = data["min"]["good"]
        # self.y_bad = data["min"]["bad"]
        self.y_good = data["max"]["bad"]
        self.y_bad = data["max"]["good"]
        self.y_optimum = data["optimum"]
        self.opt_d = data["opt_d"]

    def load(self):
        """ Ахназарова с. 207   d=exp[—(|у'|)**n  у_d = 2y- (y_max + y_min) / (y_max - y_min) ' """
        self.y1 = (2 * self.y_optimum - (self.y_good + self.y_bad)) / (self.y_good - self.y_bad)  # tested
        # self.n = (math.log(math.log(1 / self.d_param))) / (math.log(math.fabs(self.y1)))
        self.n = (math.log(math.log(1 / self.d_good))) / (math.log(math.fabs(self.y1)))
        pass

    def calc(self, x: float):  # Написано Матвеем
        self.y = (2 * x - (self.y_good + self.y_bad)) / (self.y_good - self.y_bad)
        self.d = math.exp(-(math.fabs(self.y) ** self.n))
        return self.d


class HarringtonTwoOne(Harrington):
    """Двухсторонний критерий Харрингтона"""

    def __init__(self, _subsys: Subsys = None, y_good=1, y_bad=0):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        self.type = 'one'  # one, max, min
        self.min_harrington = HarringtonOne()
        self.max_harrington = HarringtonOne()
        self.d_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
        self.d_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
        self.h_good = -math.log(math.log(1 / self.d_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / self.d_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.y_good = y_good  # Назначаем "хороший" параметр Y при self.d_good
        self.y_bad = y_bad  # Назначаем "плохой" параметр Y при self.d_bad
        self.y_optimum = 0
        self.opt_d = 0
        # self.load()  # Считаем коэффициенты в уравнении Харрингтона
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y

    def data(self, json_name):
        with open(json_name, 'r') as f:
            data = json.load(f)
        self.min_harrington.y_good = data["min"]["good"]
        self.min_harrington.y_bad = data["min"]["bad"]
        self.max_harrington.y_good = data["max"]["good"]
        self.max_harrington.y_bad = data["max"]["bad"]
        self.y_optimum = data["optimum"]
        self.opt_d = data["opt_d"]
        # imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
        # d_range_1 = []
        # for y in imt_range:
        #     if y > data["optimum"]:
        #         d = self.max_harrington.calc(y)
        #         self.type = 'max'
        #     elif y < data["optimum"]:
        #         d = self.min_harrington.calc(y)
        #         self.type = 'min'
        #     else:
        #         d = data["opt_d"]
        #         self.type = 'one'
        #     d_range_1.append(d)
        # return d_range_1

    def load(self):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.min_harrington.load()
        self.max_harrington.load()

    def calc(self, y: float):
        if y > self.y_optimum:
            self.h_level = self.max_harrington.calc(y)
            self.type = 'max'
        elif y < self.y_optimum:
            self.h_level = self.min_harrington.calc(y)
            self.type = 'min'
        else:
            self.h_level = self.opt_d
            self.type = 'one'
        return self.h_level  # Частная функция желательности d


# class HarringtonTwo:
#     """Двухсторонний критерий Харрингтона"""
#
#     def __init__(self, _subsys: Subsys = None, y_good=1, y_bad=0):
#         """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
#         self.health = _subsys  # Ссылка на родителя
#         self.type = 'one'  # one, max, min
#         self.min_harrington = HarringtonOne()
#         self.max_harrington = HarringtonOne()
#         self.d_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
#         self.d_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
#         self.h_good = -math.log(math.log(1 / self.d_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
#         self.h_bad = -math.log(math.log(1 / self.d_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
#         self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
#         self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
#         self.y_good = y_good  # Назначаем "хороший" параметр Y при self.d_good
#         self.y_bad = y_bad  # Назначаем "плохой" параметр Y при self.d_bad
#         self.load()  # Считаем коэффициенты в уравнении Харрингтона
#         self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y
#
#     def data(self):
#         with open('imt.json', 'r') as f:
#             data = json.load(f)
#         self.min_harrington.y_good = data["min"]["good"]
#         self.min_harrington.y_bad = data["min"]["bad"]
#         self.max_harrington.y_good = data["max"]["good"]
#         self.max_harrington.y_bad = data["max"]["bad"]
#         imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
#         d_range_1 = []
#         for y in imt_range:
#             if y > data["optimum"]:
#                 d = self.max_harrington.calc(y)
#                 self.type = 'max'
#             elif y < data["optimum"]:
#                 d = self.min_harrington.calc(y)
#                 self.type = 'min'
#             else:
#                 d = data["opt_d"]
#                 self.type = 'one'
#             d_range_1.append(d)
#         return d_range_1
#
#     def load(self):
#         """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
#         self.b_1 = (self.h_good - self.h_bad) / (self.y_good - self.y_bad)  # Считаем b_1 из уравнений (1) и (2)
#         self.b_0 = self.h_good - self.b_1 * self.y_good  # Считаем b_0 из уравнений (1) и (2)
#
#     def calc(self, y: float):
#         """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
#         self.h_level = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
#         return self.h_level  # Частная функция желательности d
#
#
# class Harrington1:
#     """Односторонний критерий Харрингтона """
#
#     def __init__(self, _health: Health = None):
#         """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
#         self.health = _health  # Ссылка на родителя
#         self.h_good = -math.log(math.log(1 / 0.80))  # Хороший результат по Харрингтону b_0 + b_1*y_good = h_good (1)
#         self.h_bad = -math.log(math.log(1 / 0.20))  # Плохой результат по Харрингтону b_0 + b_1 * y_bad = h_bad (2)
#         self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
#         self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
#         self.d: float = 0  # Частная функция желательности Харрингтона для параметра y
#
#     def calc(self, y_good: float, y_bad: float, y: float):
#         """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
#         self.b_1 = (self.h_good - self.h_bad) / (y_good - y_bad)  # Считаем b_1 из уравнений (1) и (2)
#         self.b_0 = self.h_good - self.b_1 * y_good  # Считаем b_0 из уравнений (1) и (2)
#         self.d = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
#         # print('h_good ', self.h_good)
#         # print('h_bad ', self.h_bad)
#         # print('b_1 ', self.b_1)
#         # print('b_0', self.b_0)
#         # print('d', self.d)
#         return self.d
#
#
# class Harrington11:
#     """Два односторонних критерия Харрингтона """
#
#     def __init__(self, _health: Health = None):
#         """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
#         self.health = _health  # Ссылка на родителя
#         self.h_good = -math.log(math.log(1 / 0.80))  # Хороший результат по Харрингтону b_0 + b_1*y_good = h_good (1)
#         self.y_good = 0  # Назначаем "хороший" параметр d = 0.8
#         self.h_bad = -math.log(math.log(1 / 0.20))  # Плохой результат по Харрингтону b_0 + b_1 * y_bad = h_bad (2)
#         self.y_bad = 0  # Назначаем "плохой" параметр d = 0.2
#         self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
#         self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
#         self.d: float = 0  # Частная функция желательности Харрингтона для параметра y
#
#     def calc(self, y_good: float, y_bad: float, y: float):
#         """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
#         self.b_1 = (self.h_good - self.h_bad) / (y_good - y_bad)  # Считаем b_1 из уравнений (1) и (2)
#         self.b_0 = self.h_good - self.b_1 * y_good  # Считаем b_0 из уравнений (1) и (2)
#         self.d = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
#         # print('h_good ', self.h_good)
#         # print('h_bad ', self.h_bad)
#         # print('b_1 ', self.b_1)
#         # print('b_0', self.b_0)
#         # print('d', self.d)
#         return self.d
#
#
class Harrington2:
    """Двухсторонний критерий Харрингтона"""

    def __init__(self, _health: Health = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.207"""
        self.health = _health  # Ссылка на родителя
        self.d_good = 0.8  # Хороший результат по Харрингтону
        self.d: float = 0  # Частная функция желательности Харрингтона для параметра y

        self.y_max = 35  # 25
        self.y_min = 15  # 18,5
        self.current_IMT = None
        self.param = 21
        self.d_param = 0.7  # 0.6
        self.y1 = None
        self.y = None
        self.n = None

    def calc(self, y: float):
        """ Ахназарова с. 207   d=exp[—(|у'|)**n  у_d = 2y- (y_max + y_min) / (y_max - y_min) ' """
        y_d = (2 * self.param - (self.y_max + self.y_min)) / (self.y_max - self.y_min)  # Считаем у_d из уравнения
        z = math.log(math.fabs(y_d))
        n = math.log(math.log(1 / self.d_param) / z)  # Считаем показатель степени
        self.d = math.exp(-math.fabs(y) ** n)  # Считаем d по Ахназаровой с.207
        return self.d

    def calc2(self, x: float):  # Написано Матвеем
        self.y1 = (2 * self.param - (self.y_max + self.y_min)) / (self.y_max - self.y_min)
        self.n = (math.log(math.log(1 / self.d_param))) / (math.log(math.fabs(self.y1)))
        self.y = (2 * x - (self.y_max + self.y_min)) / (self.y_max - self.y_min)
        self.d = math.exp(-(math.fabs(self.y) ** self.n))
        return self.d


class IMT(Subsys):
    """Управление объектами """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.health = _health  # Ссылка на родителя
        self.name = 'ИМТ'
        self.data = ''  # 'imt.json'
        self.harrington = HarringtonTwoOne()
        # self.harrington = HarringtonHalfTwo()
        self.current_value = None  # Текущее показание
        self.h_level = None  # Показатель Харрингтона

    def load(self, json_name):
        # json_name = self.__class__.__name__.lower() + '.json'
        self.data = json_name
        self.harrington.data(json_name)
        self.harrington.load()

    def calc(self, weight: float = 80, height: int = 170):
        self.current_value = weight / ((height / 100) ** 2)
        self.h_level = self.harrington.calc(self.current_value)
        return int(self.h_level * 100), round(self.current_value)


class Resp(Subsys):
    """Управление объектами """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.name = 'Дыхание'
        self.data = 'resp.json'
        self.health = _health  # Ссылка на родителя
        self.harrington = HarringtonOne()
        # self.harrington = HarringtonParabola()
        self.current_value = 33  # Текущее показание
        self.h_level = None  # Показатель Харрингтона

    def load(self, json_name):
        # json_name = self.__class__.__name__.lower() + '.json'
        self.data = json_name
        with open(json_name, 'r') as f:
            data = json.load(f)
        if self.health.user.gender == 'man':
            self.harrington.y_good = data["man"]["good"]
            self.harrington.y_bad = data["man"]["bad"]
        else:
            self.harrington.y_good = data["women"]["good"]
            self.harrington.y_bad = data["women"]["bad"]
        self.harrington.load()

    def calc(self, val: int = 30):
        self.current_value = val
        self.h_level = self.harrington.calc(self.current_value)
        return int(self.h_level * 100), self.current_value


class Heart(Resp):
    """Загрузка данных и расчет показателя пульса по Харрингтону """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.name = 'Пульс'
        self.data = 'heart.json'
        self.current_value = 74  # Текущее показание
    #     self.health = _health  # Ссылка на родителя
    #     self.harrington = HarringtonOne()
    #     self.h_level = None  # Показатель Харрингтона
    #
    # def load(self):
    #     json_name = self.__class__.__name__.lower()+'.json'
    #     with open(json_name, 'r') as f:
    #         data = json.load(f)
    #     if self.health.user.gender == 'man':
    #         self.harrington.y_good = data["man"]["good"]
    #         self.harrington.y_bad = data["man"]["bad"]
    #     else:
    #         self.harrington.y_good = data["women"]["good"]
    #         self.harrington.y_bad = data["women"]["bad"]
    #     self.harrington.load()
    #
    # def calc(self, val: int = 66):
    #     self.current_value = val
    #     self.h_level = self.harrington.calc(self.current_value)
    #     return int(self.h_level * 100), self.current_value

    # def calc(self, gender: str = 'women', age: int = 26, pulse: int = 66):
    #     df = self.df
    #     self.good_pulse = int(df.loc[(df['gender'] == gender) & (df['age'] >= age)]['good_pulse'].iloc[0])
    #     # Фильтруем по полу, возрасту и выводим первый [0] элемент серии значений как целое число
    #     self.bad_pulse = int(df.loc[(df['gender'] == gender) & (df['age'] >= age)]['bad_pulse'].iloc[0])
    #     self.current_pulse = pulse
    #     self.h_level = self.health.harrington.calc(self.good_pulse, self.bad_pulse, self.current_pulse)
    #     return self.h_level, pulse
    # print(f' gender\t{gender},\tage\t{age},\tpulse\t{pulse},\td_pulse\t{int(self.d_pulse * 100)}%')


# class Pulse(Subsys):
#     """Загрузка данных и расчет показателя пульса по Харрингтону """
#
#     def __init__(self, _health: Health = None):
#         super().__init__()
#         self.health = _health  # Ссылка на родителя
#         self.harrington = HarringtonOne()
#         self.current_value = None  # Текущее показание
#         self.h_level = None  # Показатель Харрингтона
#
#     def load(self):
#         json_name = self.__class__.__name__.lower()+'.json'
#         with open(json_name, 'r') as f:
#             data = json.load(f)
#         if self.health.user.gender == 'man':
#             self.harrington.y_good = data["man"]["good"]
#             self.harrington.y_bad = data["man"]["bad"]
#         else:
#             self.harrington.y_good = data["women"]["good"]
#             self.harrington.y_bad = data["women"]["bad"]
#         self.harrington.load()
#
#     def calc(self, val: int = 66) -> int:
#         self.current_value = val
#         self.h_level = self.harrington.calc(self.current_value)
#         return int(self.h_level * 100)


def Calibrate(json_name: str):
    """Калибровочная диаграмма"""
    har_2 = HarringtonTwoOne()
    har_2.data(json_name)
    har_2.load()
    with open(json_name, 'r') as f:
        data = json.load(f)
    imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
    d_range_1 = []
    for y in imt_range:
        d = har_2.calc(y)
        d_range_1.append(d * 100)
    plt.plot(imt_range, d_range_1, label="Калибровка", marker="o", ms=6, mfc='w')
    # plt.grid()
    plt.title(f'Калибровочная диаграмма')
    plt.ylabel('Желательность, %', loc='top', fontsize=12)  # fontweight="bold"
    plt.xlabel('Значение', loc='right', fontsize=12)
    plt.legend(loc='best')
    plt.show()
    return plt


def HarringtonShow():
    """Просмотр """
    # with open('imt.json', 'r') as f:
    #     data = json.load(f)
    #
    # imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
    # har_1 = Harrington1()
    # d_range_1 = []
    # for y in imt_range:
    #     if y > data["optimum"]:
    #         d = har_1.calc(data["max"]["good"], data["max"]["bad"], y)
    #     elif y < data["optimum"]:
    #         d = har_1.calc(data["min"]["good"], data["min"]["bad"], y)
    #     else:
    #         d = data["opt_d"]
    #     d_range_1.append(d)

    # imt_range = range(10, 65, 1)
    # imt_range = range(5, 81, 1)
    # har_1 = Harrington1()
    # y_bad_min = 10
    # y_good_min = 15
    # y_bad_max = 64
    # y_good_max = 45
    # y_optimum = 21
    # d_range_1 = []
    # for y in imt_range:
    #     if y > y_optimum:
    #         d = har_1.calc(y_good_max, y_bad_max, y)
    #     elif y < y_optimum:
    #         d = har_1.calc(y_good_min, y_bad_min, y)
    #     else:
    #         d = 0.98
    #     d_range_1.append(d)

    # print(d_range)
    # plt.plot(imt_range, d_range_1, label="two one side", marker="o", ms=6, mfc='w')
    imt_range = range(21, 81, 1)
    har_2 = Harrington2()
    d_range_2 = []
    for y in imt_range:
        d_range_2.append(har_2.calc2(y))
    plt.plot(imt_range, d_range_2, label="two side", marker="o", ms=6, mfc='w')
    # plt.grid()
    plt.title(f'Harrington1 and Harrington2')
    plt.ylabel('health part', loc='top', fontsize=12)  # fontweight="bold"
    plt.xlabel('imt', loc='right', fontsize=12)
    plt.legend(loc='best')
    plt.show()


def Half_harrington(x_beg: int, x_end: int, x: int):
    """Расчет функции желательности Харрингтона от 0 до 1"""
    # x = np.linspace(x_beg, x_end)  # Все значения исходной величины
    x01 = (((2 * x - (x_end + x_beg)) / (x_end - x_beg)) + 1) / 2  # преобразуем в безразмерную шкалу от 0 до 1
    # m = np.log(np.log(1 / 0.8)) / np.log(0.6)  # m = 3 (0.8, 0.6) - подобранные коэффициенты Харрингтона
    m = 3  # Подобранные коэффициенты Харрингтона
    k = 2  # Загиб концов графика
    d = np.exp(- (x01 * k) ** m)  # расчет функции желательности от 0 до 1
    if d > 1.0:
        d = 1.0
    return d


def Half_harrington_calibrate(x_beg: int, x_end: int):
    """Одна из калибровочной кривой ветвей Двухстороннего критерия Харрингтона"""
    # x_list = list(range(x_beg, x_end))  # Все значения исходной величины
    n = int(math.fabs(x_end - x_beg) + 1)
    x_list = list(np.linspace(x_beg, x_end, n))  # Все значения исходной величины
    # list(np.linspace(50, 45, (50 - 45 + 1))))

    d_list = []
    for x in x_list:
        d = Half_harrington(x_beg, x_end, x)
        d_list.append(d)  # Расчет функции желательности Харрингтона от 0 до 1
    return x_list, d_list
    # plt.plot(x_list, d_list, marker="o", ms=6, mfc='w')  # вывод графика функции желательности исходной величины
    # plt.xlabel('Исходный параметр', loc='right', fontsize=12)
    # plt.ylabel('Функция желательности', loc='top', fontsize=12)


def Two_harrington_calibrate(x_beg: int, x_optimum: int, x_end: int):
    """Две ветви калибровочной кривой Двухстороннего критерия Харрингтона"""
    x = []
    d = []
    x1, d1 = Half_harrington_calibrate(x_optimum, x_beg)
    x.extend(x1[::-1])
    d.extend(d1[:: -1])
    x2, d2 = Half_harrington_calibrate(x_optimum, x_end)
    x.extend(x2)
    d.extend(d2)
    plt.plot(x, d, marker="o", ms=6, mfc='w')  # вывод графика функции желательности исходной величины
    plt.xlabel('Исходный параметр', loc='right', fontsize=12)
    plt.ylabel('Функция желательности', loc='top', fontsize=12)
    plt.show()


def harr_two_one():
    # global user
    user = User()
    # user.gender = 'women'
    user.health = Health(user)
    imt = IMT()
    imt.health = user.health
    imt.load('imt.json')
    imt.calc(50, 170)
    user.health.add_subsystem(imt)
    resp = Resp()
    resp.health = user.health
    resp.load('resp.json')
    resp.calc(40)
    user.health.add_subsystem(resp)
    heart = Heart()
    heart.health = user.health
    heart.load('heart.json')
    heart.calc(60)
    user.health.add_subsystem(heart)
    values = [int(syb.h_level * 100) for syb in user.health.subsystems.values()]
    keys = [syb.name for syb in user.health.subsystems.values()]
    user.health.create_diagram(keys, values)
    for subsys in list(user.health.subsystems.values()):
        # json_file_name = subsys.__class__.__name__.lower() + '.json'
        subsys.calibrate(subsys.data, subsys.current_value, subsys.h_level * 100)


class HalfHarrington(Harrington):
    """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""

    def __init__(self, _subsys: Subsys = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        self.data = None

    def data_load(self, json_name: str):
        with open(json_name, 'r') as f:
            self.data = json.load(f)

    def calc(self, x):
        x01 = (((2 * x - (self.data['man']['end'] + self.data['man']['beg'])) /
                (self.data['man']['end'] - self.data['man']['beg'])) + 1) / 2  # в безразмерную шкалу от 0 до 1
        # m = np.log(np.log(1 / 0.8)) / np.log(0.6)  # m = 3 (0.8, 0.6) - подобранные коэффициенты Харрингтона
        m = 3  # Подобранные коэффициенты Харрингтона
        k = 2  # Загиб концов графика
        self.h_level = np.exp(- (x01 * k) ** m)  # расчет функции желательности от 0 до 1
        if self.h_level > 1.0:
            self.h_level = 1.0
        return self.h_level


if __name__ == "__main__":
    harr_two_one()  # Old

    Two_harrington_calibrate(10, 21, 80)

    # def calibre():
    #     hh = HalfHarrington()
    #     hh.data_load('resp2.json')
    #     n = int(math.fabs(hh.data['man']['beg'] - hh.data['man']['end']) + 1)
    #     x_list = list(np.linspace(hh.data['man']['beg'], hh.data['man']['end'], n))  # Значения исходной величины
    #     y_list = [hh.calc(x) for x in x_list]
    #     print(x_list, y_list)

    # calibre()

    # print(Half_harrington(20, 50, 60))
    # print(Half_harrington(50, 20, 10))
    # print(Half_harrington(50, 20, 50))
    # print(Half_harrington(20, 50, 20))
    # print(Half_harrington(50, 20, 60))
    # print(Half_harrington(20, 50, 10))
    # print(Half_harrington(20, 50, 60))

# print(keys)
# print(values)
# user = User()
# fig2 = user.health.create_diagram(['ИМТ', 'Сердце', 'Легкие'], [40, 70, 80])
# plt.show()
