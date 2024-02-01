# IE-CBIR
![image](https://github.com/M-hayatooo/IE-CBIR/assets/82699320/e787a4f5-74ff-4453-a7bd-172333d33913)


**Isometric Feature Embedding for Content-Based Image Retrieval**<br>
**Author** : [Hayato Muraki](https://github.com/M-hayatooo), Kei Nishimaki, Shuya Tobaru, Kenichi Oishi, and [Hitoshi Iyatomi](https://iyatomi-lab.info) <br>


The purpose of this research is to realize a CBIR (content-based medical image retrieval) system.<br>


**Abstract** : _Content-based image retrieval (CBIR) technology for brain MRI is needed for diagnostic support and research. To
realize practical CBIR, it is necessary to obtain a low-dimensional representation that simultaneously achieves (i) data integrity, (ii) high disease retrieval capability, and (iii) interpretability. However, conventional methods based on machine learning techniques
such as variational autoencoders (VAE) cannot acquire representations that satisfy these requirements; hence, an ad-hoc
classification model must be prepared for disease retrieval. In this paper, we propose isometric feature embedding for CBIR (IECBIR),
a low-dimensional representation acquisition framework that simultaneously satisfies the above requirements. In the
evaluation experiment using the ADNI2 dataset of t1-weighted 3D brain MRIs from 573 subjects (3,557 cases in total), the
low-dimensional representation acquired by IE-CBIR (1/4,096 of the number of elements compared with the original) achieved
a classification performance of 0.888 in F1 score and 91.5% in accuracy for Alzheimer’s disease and normal cognitive subjects,
without the need for ad hoc models, while achieving a high preservation of the original data. This diagnostic performance
outperformed machine learning methods such as CNNs (76-91% accuracy), which specialize in classification without considering
the acquisition of low-dimensional representations and their interpretability._



この研究はCBIRシステムの実現に向けたものである．高次元データな3次元脳MR画像を次元削減を行い，低次元空間に写像した後，低次元空間上で類似度計算を行い類似症例を提示することができる，画像を入力とした検索システムの開発を目標としている．<br>

低次元空間に写像したデータは入力画像が元々保持していた疾病の特徴や構造情報と差異がない事が望ましく，こうした情報を欠落させないことが必要である．従って，低次元表現から再構成した画像が入力と差異のないもの<br>であれば，低次元表現は入力の情報を欠落せず写像させていると考えることができる．ゆえに本研究では再構成画像の解像度の向上を目的とし，精度向上に取り組んでいる．


<h2> Accepted at the 58th Annual Conference on Information Sciences and Systems </h2>
This research was accepted at the CISS 2024 https://ee-ciss.princeton.edu/ <br>


本論文は CISS 2024 に採択されました。

Isometric feature embedding for content-based image retrieval , Hayato Muraki, Kei Nishimaki, Shuya Tobaru, Kenichi Oishi, and Hitoshi Iyatomi, Proc. Information Sciences and Systems  (CISS2024), Mar. 2024. (accepted)







  
<h2>Version</h2>
  
  <table>
<thead>
<tr>
<th>Library</th>
<th>Version</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>Python</code></td>
<td><code>3.6 (Anaconda)</code></td>
</tr>
<tr>
<td><code>torch</code></td>
<td>&gt;= <code>1.2</code> (tested on <code>1.7</code>)</td>
</tr>
<tr>
<td><code>torchvision</code></td>
<td>&gt;= <code>0.4</code></td>
</tr>
<tr>
<td><code>matplotlib</code></td>
<td>&gt;= <code>2.2.2</code></td>
</tr>
<tr>
<td><code>numpy</code></td>
<td>&gt;= <code>1.17</code></td>
</tr>
<tr>
<td><code>opencv</code></td>
<td>&gt;= <code>3.4.2</code></td>
</tr>
<tr>
<td><code>tqdm</code></td>
<td>&gt;= <code>4.36.1</code></td>
</tr>
<tr>
<td><code>scipy</code></td>
<td>&gt;= <code>1.3.1</code></td>
</tr>
</tbody>
</table>

