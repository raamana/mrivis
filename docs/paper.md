---
title: 'medical image visualization library for neuroscience in python'
tags:
  - visualization
  - neuroscience
  - alignment
  - histogram
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

Neuroscience research routinely demands a variety of visualization tasks, ranging from a simple 2D image to custom-built composite stacks. While different laboratories attend to this need differently, among which the majority of users tend to use what's already available even if its not ideal. While few laboraries, when resources and skills permit, engage in in-house software development to temporarily solve it, they are not often either open source at all, or made with the intent to be distributed widely and to be reliable. We aim to address this need with fully-open-source and pure-python implementaion.

mrivis provides easy ways to perform non-trivial medical image visualization tasks, such as visual comparison of sptial alignment of neuroimaging data. In addition, we provide a base development kit containing carefully-deisgned python classes to traverse through 3D neuroimaging data (`SlicePicker`), produce customizable collages (`Collage`) and to flatten 4D or higher-dimensional MRI data into 2D images (`Carpet`). These classes together form a easy to use development kit to build even more customized visualizations, which is often needed for cutting-edge neuroscience research.

It is based on matplotlib [@hunter2007matplotlib], nibabel [@brett2016nibabel] and numpy [@oliphant2007python], and is already serving visualqc [@raamana2018visualqc].

