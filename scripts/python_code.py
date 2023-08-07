import numpy as np
from PIL import Image
from wordcloud import WordCloud

char_mask = np.array(Image.open("~/Document/Project_KMUTNB/Pictures/circle.png"))

wordcloud = WordCloud(background_color="black",mask=char_mask).denerate(text)

plt.figure(figsize = (8,8))
plt,imshow(wc)

plt.axis("off")
plt.show()

