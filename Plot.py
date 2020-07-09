import matplotlib.pyplot as plt
import seaborn as sns


class Plot:
    def __init__(self, classes):
        # define possible colors for each class
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkblue', 'deeppink', 'dimgray', 'ivory', 'indigo', 'lime',
                'magenta', 'maroon', 'mintcream', 'navy', 'olive', 'oldlace', 'pink', 'peru', 'powderblue', 'rosybrown']

        # create a dictionary of classes each with its color
        self.cls_clr = {}
        for i in range(len(classes)):
            self.cls_clr[classes[i]] = clrs[i]

        self.fig, self.ax = plt.subplots()

    def show_plot(self, x_axis, y_axis, num, dataset):
        for i in range(num):
            d = dataset['class'][i]
            c = self.cls_clr[d]
            self.ax.scatter(dataset[x_axis][i], dataset[y_axis][i], color=c)

        # set a title and labels
        self.ax.set_title('Plotting result')
        self.ax.set_xlabel(str(x_axis))
        self.ax.set_ylabel(str(y_axis))

        plt.show()

    def visulaize(self, dataset, x_axis, y_axis):
        sns.set(color_codes=True)
        sns.FacetGrid(dataset, hue="class", palette="husl", size=5) \
            .map(plt.scatter, x_axis, y_axis) \
            .add_legend()


"""
####################
EXAMPLE DRIVER CLASS
####################

# load dataset
# names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'class']
# dataset = pd.read_csv(filename.csv, names=[feature names])
dataset = pd.read_csv('wine.csv', names=['x1', 'x2', 'class'])
# print(dataset.head())
############################################################################################
# Extract labels from dataset
lbls = dataset['class'].unique()
# print(lbls)
############################################################################################
# Shuffle dataset
# frac=1 means return all rows (in random order).

dataset_shuffle = dataset.sample(frac=1).reset_index(drop=True)
# print(dataset_shuffle.head())
############################################################################################

# Create Plot object
# plt.show_plt(feature1_for_x-axis, feature2_for_y-axis, number_of_samples, shuffled_dataset)

plt = Plot(lbls)
plt.show_plot('x1', 'x2', 50, dataset_shuffle)

"""
