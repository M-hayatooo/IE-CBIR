# IE-CBIR
Isometric Feature Embedding for Content-Based Image Retrieval

The purpose of this research is to realize a CBIR (content-based medical image retrieval) system.<br>
この研究はCBIRシステムの実現に向けたもの.

高次元データな3次元脳MR画像を次元削減を行い，低次元空間に写像した後，低次元空間上で類似度計算を行い<br>類似症例を提示することができる画像を入力とした検索システムの開発を目標としている．<br><br>

低次元空間に写像したデータは入力画像が元々保持していた疾病の特徴や構造情報と差異がない事が望ましく，<br>こうした情報を欠落させないことが必要である．
従って，低次元表現から再構成した画像が入力と差異のないもの<br>であれば，低次元表現は入力の情報を欠落せず写像させていると考えることができる．<br>
ゆえに本研究では再構成画像の解像度の向上を目的とし，精度向上に取り組んでいる．


<h2> Accepted at the 58th Annual Conference on Information Sciences and Systems </h2>
This research was accepted at the CISS 2024 https://ee-ciss.princeton.edu/ <br>


本論文は CISS 2024 に採択されました。

Isometric feature embedding for content-based image retrieval , Hayato Muraki, Kei Nishimaki, Shuya Tobaru, Kenichi Oishi, and Hitoshi Iyatomi, Proc. Information Sciences and Systems  (CISS2024), Mar. 2024. (accepted)







  
Version
  
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

