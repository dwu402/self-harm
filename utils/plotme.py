import matplotlib.pyplot as plt
import click
import numpy as np
import re
from cycler import cycler

new_color = cycler(color=["k"])
plt.rcParams['axes.prop_cycle'] = plt.rcParams['axes.prop_cycle'].concat(new_color)

@click.command()
@click.option('-f','--filename',help='file to read in')
def main(filename):
	def flts(x=[-1]):
		x[0] = x[0] + 1
		return r"(?P<float_{}>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)".format(x[0])
	raw_rxp = flts() + r"\s*\(" + flts() + r"\s*,\s*" + flts() + r"\)\s*\|"+ "".join(r"\s*"+flts() for _ in range(11))	
	rxp = re.compile(raw_rxp)
	parse_results = []
	with open(filename,'r') as f:
		line = f.readline()
		while line:
			matched = rxp.match(line)
			parse_results.append(matched)
			line = f.readline()
	values = [[float(res.group("float_{}".format(i))) for i in range(14)] for res in parse_results]
	mice = [val[0] for val in values]
	alphas = [val[1] for val in values]
	rhos = [val[2] for val in values]
	pars = [val[3:] for val in values]
	mice_ns = range(len(mice))

	plt.plot(mice_ns, pars, 'o--', linewidth=0.25)
	plt.legend("rkpmscdfgjl", loc="best", bbox_to_anchor=(1.01, 1))
	plt.show()	

if __name__ == "__main__":
	main()
