# ArtRecommendation
This application is an art recommendation ML model that will grab art pieces chosen by the user and will recommend other art pieces that the user could also enjoy

## Museum Repos:
  https://github.com/metmuseum/ (limited to 80 requests per second) <br />
  https://data.rijksmuseum.nl/object-metadata/api/

## Overall Plan:
  1. Get images from Met API of Artwork
  2. Use ML model (most likely in Python) to identify patterns in the images (Image recognition and classification)
  3. See if the model can recommend something with a control of all images from 2 artists and see if me selecting 3 of one type of artist will recommend a fourth one by said artist
  4. Import even more images and check accuracy at scale
  5. Present x amount of random paintings and have users select which ones they like and then present an artwork they could like
  6. Implement reshuffle of non-selected art pieces in random presentation
