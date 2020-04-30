import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


# draw the picture
def draw(Curve_one, Curve_two, Curve_three, Curve_four):
    plt.figure()

    plot1, = plt.plot(Curve_one[0], Curve_one[1], 'co-', linewidth=1.0, markersize=5.0,color='orangered')
    plot2, = plt.plot(Curve_two[0], Curve_two[1], 'bx-', linewidth=1.0, markersize=5.0,color='olivedrab')
    plot3, = plt.plot(Curve_three[0], Curve_three[1], 'rh-', linewidth=1.0, markersize=5.0,color='steelblue')
    plot4, = plt.plot(Curve_four[0], Curve_four[1], 'k^-', linewidth=1.0, markersize=5.0,color='orchid')

    # set X axis
    plt.xlim([0, 1.05])
    plt.xticks(np.linspace(0, 1.0, 11))
    plt.xlabel("Recall", fontsize="x-large")

    # set Y axis
    plt.ylim([0, 1.05])
    plt.yticks(np.linspace(0, 1.0, 6))
    plt.ylabel("Precision", fontsize="x-large")

    # set figure information
    # plt.title("Precision-Recall Curve", fontsize="x-large")
    plt.legend([plot1, plot2, plot3, plot4], ("NTFEM-W", "NTFEM-UW", "NTFEM-BiTree-OPT", "NTFEM-BiTree-SLT"), loc="upper right",
               numpoints=1)
    plt.grid(linestyle='--')
    plt.savefig("line_f1.eps", format="eps")

    # draw the chart
    # plt.show()


# main function
def main():
    # Curve one main curve
    # Curve_one = [
    #     (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    #     (0.9518,0.6261,0.4542,0.4166,0.3076,0.2639,0.1986,0.1549,0.0839,0.0425,0.0000)
    # ]
    #
    # # Curve two without tree-sif unweighted
    # Curve_two = [
    #     (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    #     (0.9571,0.5285,0.3338,0.2752,0.2173,0.1725,0.1152,0.0773,0.0408,0.0149,0.0000)]
    #
    # # Curve three symbol with binary opt structure
    # Curve_three = [
    #     (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    #     (0.9515,0.4991,0.3598,0.2729,0.2273,0.1423,0.0843,0.0465,0.0145,0.0017,0.0000)]
    #
    # # Curve with binary SLT tree symbol
    # Curve_four = [
    #     (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    #     (0.9539,0.5454,0.3644,0.3162,0.2472,0.1604,0.0875,0.0547,0.0238,0.0073,0.0000)]
    # #
    # Curve one main curve
    Curve_one = [
        (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        (0.8991, 0.8163, 0.7679, 0.5912, 0.5113, 0.4218, 0.3741, 0.3678, 0.2660, 0.2518, 0.2361)
        ]

    # Curve two without tree-sif unweighted
    Curve_two = [
        (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        (0.8959,0.8109,0.6601,0.4896,0.4007,0.3972,0.3246,0.2932,0.1936,0.1925,0.1884)]

    # Curve three symbol with binary opt structure
    Curve_three = [
        (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        (0.8904,0.8010,0.6006,0.4422,0.3861,0.3593,0.3090,0.2984,0.1871,0.1829,0.1738)]

    # Curve with binary SLT tree symbol
    Curve_four = [
        (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        (0.8959, 0.7514, 0.6055, 0.4932, 0.4054, 0.3795, 0.3418, 0.3316, 0.2332, 0.2303, 0.2275)]

    # Call the draw function
    draw(Curve_one, Curve_two, Curve_three, Curve_four)


# function entrance
if __name__ == "__main__":
    main()
