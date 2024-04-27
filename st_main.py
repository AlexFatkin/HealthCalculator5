import streamlit as st

import health_sys as hs

hs.view_console = False  # Не выводить графику в консоль
main_user = hs.User()
main_user.health = hs.Health(main_user)

gender_index = 0  # women
if main_user.gender == "women":
    gender_index = 0
else:
    gender_index = 1

if "user" not in st.session_state:
    st.session_state.user = main_user
    st.session_state.disabled = False

st.title("Калькулятор здоровья")
with st.sidebar:
    page = st.radio("Подсистема организма", ("Легкие", "Сердце", "Жировой запас"))

hs.view_console = False


def set_gender():
    """Фиксация индекса пола """
    global gender_index
    if gender == 'women':
        gender_index = 0
    else:
        gender_index = 1
    st.session_state.user.gender = gender


def show(subs):
    """Показываем графику в браузере"""
    st.session_state.user.health.add_subsystem(subs)
    values = [int(syb.h_level * 100) for syb in st.session_state.user.health.subsystems.values()]
    keys = [syb.name for syb in st.session_state.user.health.subsystems.values()]
    plt = st.session_state.user.health.create_diagram(keys, values)
    st.write(f'Диаграмма функции желательности')
    st.pyplot(plt.gcf())
    plt.close()
    st.write(f'Калибровочная диаграмма')
    plt2 = subs.calibrate(subs.data, subs.current_value, subs.h_level * 100)
    st.pyplot(plt2.gcf())


if page == "Жировой запас":
    st.header("""Индекс массы тела (ИМТ)""")
    st.text("Для расчета индекса массы тела введите свой:")
    weight = st.number_input(' вес в килограммах', value=90, placeholder="Вес в кг")
    height = st.number_input(' рост в сантиметрах', value=170, placeholder="Рост в см")

    if st.button('Рассчитать функцию желательности '):
        imt = hs.IMT()  # Создаем объект Subsys,
        imt.health = st.session_state.user.health  # добавляем в него общий по сессии объект Health
        imt.load('imt.json')  # ctrl-q for a quick-doc of the function under the cursor.
        imt.calc(weight=weight, height=height)
        show(imt)
        # st.write(f'Функция желательности {imt.name} = {h_level}%, \t   {imt.name} = {value}')


elif page == "Сердце":
    st.header("""Сердце:""")
    st.text("Для расчета индекса пульса измерьте свой пульс в покое (ударов в минуту):")
    input_value = st.number_input(' введите свой пульс в поле', value=74,
                                  placeholder="Пульс а покое")
    gender = st.selectbox(' введите свой пол', ('women', 'man'), index=gender_index)
    set_gender()
    if st.button('Рассчитать функцию желательности'):
        heart = hs.Heart()
        heart.health = st.session_state.user.health
        heart.load('heart.json')
        heart.calc(input_value)
        show(heart)


elif page == "Легкие":
    st.header("""Легкие:""")
    st.text("Для расчета индекса легких измерьте задержку дыхания в секундах:")
    input_value = st.number_input(' введите задержку дыхания в секундах в поле', value=55,
                                  placeholder="Задержка дыхания в секундах")
    gender = st.selectbox(' введите свой пол', ('women', 'man'), index=gender_index)
    set_gender()
    if st.button('Рассчитать функцию желательности'):
        resp = hs.Resp()
        resp.health = st.session_state.user.health
        resp.load('resp.json')
        resp.calc(input_value)
        show(resp)

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