import matplotlib.pyplot as plt
import pandas as pd 
import os


def plot_graph(filename, title, xlabel, ylabel):
	
	outfile = "images"+os.sep+title.replace(" ", "_")+".png"

	df = pd.read_csv(filename, encoding='utf-8')
	ax = plt.gca()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	df.plot(kind='line',x=xlabel, y='Python', color='blue', ax=ax, title=title)
	df.plot(kind='line',x=xlabel,y='Scala', color='red', ax=ax)
	plt.savefig(outfile)
	plt.clf()
	plt.cla()
	plt.close()
	print("Plot stored in: "+outfile)


if __name__ == '__main__':
	
	plot_graph("dimensions/load_time.csv", "Dimension Vs Load time", "Dimension", "Load time (s)")
	plot_graph("dimensions/run_time.csv", "Dimension Vs Run time", "Dimension", "Run time (s)")
	plot_graph("dimensions/memory_usage.csv", "Dimension Vs Memory usage", "Dimension", "Memory usage (MB)")

	plot_graph("sample_size/load_time.csv", "Sample size Vs Load time", "sample_size", "Load time (s)")
	plot_graph("sample_size/run_time.csv", "Sample size Vs Run time", "sample_size", "Run time (s)")
	plot_graph("sample_size/memory_usage.csv", "Sample size Vs Memory usage", "sample_size", "Memory usage (MB)")


