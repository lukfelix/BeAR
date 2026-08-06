/*
* This file is part of the BeAR code (https://github.com/newstrangeworlds/BeAR).
* Copyright (C) 2024 Daniel Kitzmann
*
* BeAR is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* BeAR is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You find a copy of the GNU General Public License in the main
* BeAR directory under <LICENSE>. If not, see
* <http://www.gnu.org/licenses/>.
*/
#include "sampled_data.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <assert.h>
#include <cmath>


namespace bear{

void CrossSectionFile::loadFile()
{
  std::fstream file;


  file.open(filename.c_str(), std::ios::binary | std::ios::in | std::ios::ate);

  if (file.fail()) std::cout << "cross section file " << filename << " not found! :(((( \n";
  
  assert (!file.fail( ));


  int nb_data_points = file.tellg()/sizeof(float);
  file.seekg(0, std::ios::beg);

  
  //read the entire file in one go, rather than one float at a time
  std::vector<float> buffer(nb_data_points);

  file.read((char *) buffer.data(), nb_data_points * sizeof(float));

  if (file.gcount() != static_cast<std::streamsize>(nb_data_points * sizeof(float)))
  {
    std::cout << "cross section file " << filename << " could not be fully read! :(((( \n";

    assert (false);
  }


  cross_sections.resize(nb_data_points);

  for (int i=0; i<nb_data_points; ++i)
    cross_sections[i] = buffer[i];

  if (is_data_log)
    for (int i=0; i<nb_data_points; ++i)
      cross_sections[i] = std::pow(10.0, cross_sections[i]);


  file.close();

  is_loaded = true;
}


void CrossSectionFile::unloadData()
{
  std::vector<double>().swap (cross_sections);

  is_loaded = false;
}


}