The annotations are included as a `jsonl` file.

Images can be downloaded using the provided links (`left_url` and `right_url`). Since links availability is not
guaranteed, we also provide [a file containing all images](https://drive.google.com/open?id=1QskOzSiRhlmZeMk1-8C80QHyRmW1S0zj).
The name of the image file for each example is in the following format: `{identifier}-img{0/1}.{png/jpg}` where `0/1` 
is for left/right images respectively. 

Example question:

```
{
  "sentence": "An image shows a roll of paper towels on a vertical stand with a wooden part sticking out of the top.", 
  "left_url": "https://i.pinimg.com/736x/ca/fc/d9/cafcd9dadf0ca3057444ec732ec7729f--paper-towels-toilet-paper.jpg", 
  "right_url": "https://s-media-cache-ak0.pinimg.com/originals/30/b7/38/30b738c6e4aba44a11531d7396c968e9.jpg", 
  "identifier": "test1-769-1-0-1", 
  "writer": 0,
  "label": "False"
}
```