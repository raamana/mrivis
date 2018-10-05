---
title: 'mrivis: Medical image visualization library for neuroscience in python'
tags:
  - visualization
  - neuroscience
  - alignment
  - neuroimaging
  - multimodal
  - development kit
authors:
 - name: Pradeep Reddy Raamana
   orcid: 0000-0003-4662-0558
   affiliation: 1
 - name: Stephen C. Strother
   orcid: 0000-0002-3198-217X
   affiliation: 1, 2
affiliations:
 - name: Rotman Research Institute, Baycrest Health Sciences, Toronto, ON, Canada
   index: 1
 - name: Department of Medical Biophysics, University of Toronto, Toronto, ON, Canada
   index: 2
date: 3 August 2018
doi: 10.5281/zenodo.1328279
bibliography: paper.bib
---

# Summary

Neuroscience research routinely demands a variety of visualization tasks, ranging from a simple 2D image to custom-built composite stacks. Different academic laboratories attend to this need differently, from being users of existing solutions to being developers of new software. The majority of them tend to be mostly users of what's already available, even though the current solutions are suboptimal or inefficient for the task at their hand. Some laboratories, when resources and skills permit, engage in in-house software development to try solve their problem. The resulting software are often either not open source at all, nor made with the intent to be reliable or distributed widely. We aim to address this need with a fully-open-source and pure-python visualization library.

`mrivis` provides easy ways to perform non-trivial medical image visualization tasks, such as visual comparison of spatial alignment of neuroimaging data. In addition, we provide a base development kit containing the following carefully-designed python classes
 - to traverse through 3D neuroimaging data (`SlicePicker`),
 - produce customizable collages (`Collage`) and
 - to flatten 4D or higher-dimensional MRI data into 2D images (`Carpet`) [@power2017carpet].

These classes together form an easy to use development kit to build even more customized visualizations, which is often needed for cutting-edge neuroscience research.

`mrivis` is dependent on the following libraries: `matplotlib` [@hunter2007matplotlib], `nibabel` [@brett2016nibabel] and `numpy` [@oliphant2007python], and is already serving `visualqc` [@raamana2018visualqc].

# Acknowledgement

Pradeep Reddy Raamana is grateful for the support of the Canadian Biomarker Integration Network for Depression (CAN-BIND) and Ontario Neurodegenerative Disease Research Initiative (ONDRI), which are two integrated discovery programs of the Ontario Brain Institute (OBI), Canada. OBI is an independent non-profit corporation, funded partially by the Ontario government. The opinions, results, and conclusions are those of the authors and no endorsement by the OBI is intended or should be inferred.

# References
