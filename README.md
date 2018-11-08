# Category-Prediction
Implement Convolutional Neural Network (CNN) in Python with tensorflow to predict picture category

The goals:

•	To understand the steps to train/test the model for image classification.

•	Understand architecture of CNN and how to connect each layer together by using TensorFlow.

This project build a Convolutional Network classifier using these packages: Tensorflow, Keras (for CIFAR-10), Numpy, and OpenCV (for reading image). The data set used in this project will be CIFAR-10. The common used classifiers are SVM and Softmax.

First Creat folder "./model"

Use command "python CNNclassify.py train" to train the model, save the model in a folder named “model” after finish the training.
![Fig. 1 the screenshot of training result](https://lh3.googleusercontent.com/f85oAj_x5zYOa98zl1FaR_B1Gk5F4PYnQ1o11HbRmBrHSnUNfT0olGW-LMWst8oXM4fSNDcJ0joWHKYoR8O5SsrplwmgBJF6o__5i3Qk28JV5WKy2v_3IYTNLfGmchh9rpu4kb8tay8nvyG_D6-MaXpFnNnBeaH26nQujqCHpQ0TXgmPNkBo-iFX3hCfP15Kkuw7GNQ3xC7X8zktdHzMshUzJxQ-EhVEgSHd6ANIkTTHMDENzCw_kOC6teFP3kRnstJhx47t5TemE0F_InWQz72yvocJY1NFj1MC_9JFxOGX1AuaixQIaXAPc9ZYA0gH7kLu07wraNlMFl4W5WBNwoUVDGqMgVpsNM68NTn6A0b2IeYFxRHEWir4xnfUcRYwiyTRuyIawLYAIPWiitMWRdPMJTBNaOkf-GZB6bmelQsEHCrdLqDNlKtt6Wc9O9NtbUtyg8N3xO9uUQ_PWPsbxJchuyhmt3gxx_RhjvGKcp9tY3QBoyUR94ihyLbJMFIWTDgd0Jd4n6ma_iAZ5K7bTCqGo9ejrNxxetboVyobl5XNEWyXY_6QMbSV64h5-GbydqigSK_AhqUVpYkLFW2JrdA8lcE8Y5-Gf8pcLKPAphp2Gp1u6oB1_vRyJHX5XG5fsaV4WDnmSviuaupEiqCC62OxrzLDnTCkcQdvNdLCRxrBtg=w742-h314-no)
Use command "python CNNclassify.py test xxx.png" to test the model. This command will (i) load your model from your folder “model” in the previous step. And it will read “xxx.png” and predict the output, and (2) visualize the output of first CONV layer of the trained model for each filter (e.g., 32 visualization results), and save the visualization results as “CONV_rslt.png” as shown in Fig. 3.
The testing result would match your image type when the classifier achieves high accuracy.
![Fig. 2 the screenshot of testing result](https://lh3.googleusercontent.com/otWK6mRRzkPbDC_tLxbF0KUfp8CHri-xAh7PrX7qXzx8aLq6uQiAgzc5uquhqP-neHqR7aaoYnbxyWE1-NfzG5wJ7VbawuvGqhJUe3Lw4YPxplxWVYfqxMmQ9hXxMcZThHrsmTAx_BZPa27dOr5T5GSNv4Yb9lnKoyaUrlMsHkZAZJdCcILGhyZb5TJSekO3cLNZj5p0GzpaBg9r6MXlkNbmPp3tn1rytmN1SE7Uurddbwilwwc3fJCV6eeYl2fEy4C_QdjXqTdOrYHjyZ8mCz3-bQMGX5q9Fac2lHHS-rDqzyrnwZJK4oeClD-AoI-kl0W-e3dNz6Q-EqTQN0WQhXpfe9lambHhV7fdOYJKJJDDp8fmVPv3yA4uFmUReHYuvCplGtZGqb1cSkg3ltshrZtXzaP04boJMF-nHayWRXzkIMj-sBGcuN13MpUe2arfHdK4H5EEWqOdecXoqkRD9uSrTlywuj_PZqa-6EZN8taQRFybAf1NIULcGCWPs3_SNKNAgLqWmlCpJY1bu0k6BMAkA8fLxMkW9UDblYAJJMKHTKpw3uwIdB2iLX8BdmzhsXBmWayWuDToJ1Hctd9za-ZUKJY5yrg1cfsKEsKbU41imBjEOP4Wvb-Y7VY3AXkk-GChiqKtN6C5gbfdb98opcWWd_UbcBmRlQ-8m5mdfdp3kA=w865-h89-no)
![Fig. 3 the screenshot of the first CONV layer visualization (car shape)](https://lh3.googleusercontent.com/nywH-YiApXS65a2UhPEjpPl__FnEayXG3NONidUa8U2Nba50GDGaAqd4z1AE5v3gp1nkaWak3h_CyVhm9Rv4lqwrAdPx8cUSzYLp3q1HQlTqj0zLCdwOE2CxmgQelBJp5vl7VtYqcDA8KjddvfPCUCxtyrMrt2VwP_sJu685j_5vzk8aJTJqAkQTYx-emw8vDMVPWcyZAHn4eFTrabLmFq0IpyJ9rp8Qqq9g7NW8_8fq17ELMFi4xiioGXf1yqlQeUYDf1I8Cs9hC8LxzrhOZESpPKU4uvrUSpOgtBDjdEvPPz8Y5mg_q-ZySjxYqjrM557Hkv91LRtY2nF-7DcqWXjx4nVGn8MCcOkAXW5Jv_wdw0kzYXP6exiJZg9GljJ7OY26iDlWJDFj-OxfE54tXzvgiwRajWhuuU2aek6jj6pGDK3ccVQ8hcF8K4L-ZZZ5co2gvQZEsb2y8NWHSSGiOy4B-sLWWlI7_j2zotcenvFHGdYUKwDAwxJ7UtW8LkttR4JAr5gJYC0tMm8_ZDrs843pg5otrtGR-Rtz6tjTX1l9Ic6KUj0ulFsrXDZzXbYHcuLzrt5OvLiO3tR4rt7cuzHOOemY1A8ua6yGFGw52vX8mtcTM9c3cOvab92UP9N7xezOc0vtxzzMqS5tQvnnltMjHZkRF2COC0phPpMeeV3bPA=w640-h480-no)

*Please be aware that the computational cost of CONV layer is very high and the training process may take quite long. 
