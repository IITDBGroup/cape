plots := crime_num_att.pdf crime_size.pdf crime_fd_on_off.pdf dblp_size.pdf \
			crime_bar.pdf expl_DBLP_numpat.pdf expl_crime_numatt.pdf expl_crime_numpat.pdf

all: $(plots)

%.pdf:%.py
	python3 $<
