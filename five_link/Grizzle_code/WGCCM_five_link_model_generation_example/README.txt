The archive WGCCM_five_link_model_generation_example contains MATLAB
scripts to generate the equations of motion for a 5-link (kneed) biped
walker with the morphology of RABBIT.

The files contained in this archive are associated with the book
Feedback Control of Dynamic Bipedal Robot Locomotion by Eric
R. Westervelt, Jessy W. Grizzle, Christine Chevallereau, Jun-Ho Choi,
and Benjamin Morris published by Taylor & Francis/CRC Press in 2007.

Copyright (c) 2007 by Eric R. Westervelt, Jessy W. Grizzle, Christine
Chevallereau, Jun-Ho Choi, and Benjamin Morris.  This code may be
freely used for noncommercial ends.  If use of this code in part or in
whole results in publication, proper citation must be included in that
publication.  This code comes with no guarantees or support.

To run the walking simulation execute the command

    generate_5_link_model

with no arguments at the MATLAB command prompt.  The parameters required by 
the routines D_matrix, C_matrix, and G_vector are specified on line 241 of
generate_5_link_model.m.

The scripts contained in this archive are known to work with versions
of MATLAB up to Version 7.3 (R2006b).

Eric Westervelt
20 February 2007
