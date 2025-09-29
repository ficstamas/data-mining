import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


def plot_coord_system(vectors: list, colors: list, text: list):
    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    ticks_frequency = 1

    # vectors
    xs = vectors[:, 0]
    ys = vectors[:, 1]

    # Plot points
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(xs, ys, c=colors)

    for i in range(len(xs)):
        c = colors[i]
        if c == "r":
            p = {"label": "hashed values"}
        elif c == "g":
            p = {"label": "input vector"}
        elif c == "m":
            p = {"label": "hash function"}
        else:
            p = {}
        ax.scatter(xs[i], ys[i], c=colors[i], **p)

    # Draw lines connecting points to axes
    for x, y, c, txt in zip(xs, ys, colors, text):
        ax.plot([0, x], [0, y], c=c, ls='--', lw=1.5, alpha=0.5)
        ax.annotate(txt, (
            x + 0.03 if np.sign(x) == 1 else x - 0.13,
            y + 0.03 if np.sign(y) == 1 else y - 0.13
        ), color=c, size=16)

    # Set identical scales for both axes
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=16, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=16, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels, label_set = [], [], set()
    for h, l in zip(handles, labels):
        if l in label_set:
            continue
        new_handles.append(h)
        new_labels.append(l)
        label_set.add(l)

    ax.legend(handles=new_handles, labels=new_labels, loc="lower left", prop={'size': 16})
    return fig, ax


def rotate_vector(vector, angle):
    x, y = vector
    theta = angle * np.pi / 180;

    cs = np.cos(theta);
    sn = np.sin(theta);
    px = x * cs - y * sn;
    py = x * sn + y * cs;
    return [px, py]


def plot_hyperplane_step_1(angle=30):
    vectors = np.array([
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1]
    ])
    colors = [
        'r', 'r', 'r', 'r'
    ]
    text = [
        '$A$', '$B$', '$C$', '$D$'
    ]
    fig, ax = plot_coord_system(vectors, colors, text)


def plot_hyperplane_step_2(angle=30):
    vectors = np.array([
        [0.1, 1.2],
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1]
    ])
    colors = ['g', 'r', 'r', 'r', 'r']
    text = ['$X$', '$A$', '$B$', '$C$', '$D$']

    fig, ax = plot_coord_system(vectors, colors, text)


def plot_hyperplane_step_3(angle=30):
    a = [0.393919, 0.919145]
    vectors = np.array([
        [0.1, 1.2],
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1],
        a,  # [-0.8, 0.6]
    ])
    colors = ['g', 'r', 'r', 'r', 'r', 'm']
    text = [
        '$X$', '$A$', '$B$', '$C$', '$D$', '$h_1$',  # '$h_2$'
    ]

    fig, ax = plot_coord_system(vectors, colors, text)


def plot_hyperplane_step_4(angle=30):
    a = [0.393919, 0.919145]
    vectors = np.array([
        [0.1, 1.2],
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1],
        a,  # [-0.8, 0.6]
    ])
    colors = ['g', 'r', 'r', 'r', 'r', 'm']
    text = [
        '$X$', '$A$', '$B$', '$C$', '$D$', '$h_1$',  # '$h_2$'
    ]

    hyperplanes = vectors[-1:]

    fig, ax = plot_coord_system(vectors, colors, text)

    # draw hyper planes
    for i, (x, y) in enumerate(zip(hyperplanes[:, 0], hyperplanes[:, 1])):
        ax.axline((y, -x), (-y, x), c='m')


def plot_hyperplane_step_5(angle=30):
    a = [0.393919, 0.919145]
    vectors = np.array([
        [0.1, 1.2],
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1],
        a,  # [-0.8, 0.6]
    ])
    colors = ['g', 'r', 'r', 'r', 'r', 'm']
    text = [
        '$X$', '$A$', '$B$', '$C$', '$D$', '$h_1$',  # '$h_2$'
    ]

    hyperplanes = vectors[-1:]

    fig, ax = plot_coord_system(vectors, colors, text)

    ax.annotate("$+ (= 1)$", (1.40, -0.6), color="black", size="23")
    ax.annotate("$- (= 0)$", (1.40, -1.0), color="black", size="23")

    # draw hyper planes
    for i, (x, y) in enumerate(zip(hyperplanes[:, 0], hyperplanes[:, 1])):
        ax.axline((y, -x), (-y, x), c='m')
        ax.fill_between(np.linspace(-20 * y, 20 * y, 10), -np.linspace(-20 * x, 20 * x, 10), 10, color='yellow',
                        alpha=0.3, zorder=-1)


def plot_hyperplane_step_6(angle=30):
    a = [0.393919, 0.919145]
    b = rotate_vector(a, angle)
    vectors = np.array([
        [0.1, 1.2],
        [-0.1, 0.5], [0.3, 0.2], [-0.4, -1.4], [1.3, -0.1],
        a, b
    ])
    colors = ['g', 'r', 'r', 'r', 'r', 'm', 'm']
    text = [
        '$X$', '$A$', '$B$', '$C$', '$D$', '$h_1$', '$h_2$'
    ]

    hyperplanes = vectors[-2:]

    fig, ax = plot_coord_system(vectors, colors, text)

    # ax.annotate("$3\ |\ 11$", (1.5, 0.1), color="black", size="23")
    # ax.annotate("$2\ |\ 10$", (-0.5, 1.7), color="black", size="23")
    # ax.annotate("$1\ |\ 01$", (0.1, -1.8), color="black", size="23")
    # ax.annotate("$0\ |\ 00$", (-1.9, -0.2), color="black", size="23")

    # draw hyper planes
    for i, (x, y) in enumerate(zip(hyperplanes[:, 0], hyperplanes[:, 1])):
        ax.axline((y, -x), (-y, x), c='m')
        ax.fill_between(np.linspace(-2000 * y, 2000 * y, 10000), -np.linspace(-2000 * x, 2000 * x, 10000),
                        np.sign(y) * 10, color='yellow' if i == 0 else 'cyan', alpha=0.3, zorder=-1)


@interact(
    step=widgets.ToggleButtons(
        options=[1, 2, 3, 4, 5, 6],
        description='Step:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        # tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
        # icons=['check'] * 3
    ),
    angle=widgets.IntSlider(min=1, max=359, step=1, value=220),
)
def plot_hyperplane_example(step, angle):
    eval(f"plot_hyperplane_step_{step}({angle})")