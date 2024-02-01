# IE-CBIR
Isometric Feature Embedding for Content-Based Image Retrieval

The purpose of this research is to realize a CBIR (content-based medical image retrieval) system.<br>
この研究はCBIRシステムの実現に向けたもの.

高次元データな3次元脳MR画像を次元削減を行い，低次元空間に写像した後，低次元空間上で類似度計算を行い<br>類似症例を提示することができる画像を入力とした検索システムの開発を目標としている．<br><br>

低次元空間に写像したデータは入力画像が元々保持していた疾病の特徴や構造情報と差異がない事が望ましく，<br>こうした情報を欠落させないことが必要である．
従って，低次元表現から再構成した画像が入力と差異のないもの<br>であれば，低次元表現は入力の情報を欠落せず写像させていると考えることができる．<br>
ゆえに本研究では再構成画像の解像度の向上を目的とし，精度向上に取り組んでいる．


CISS 2024 https://ee-ciss.princeton.edu/
accept




<h2> 低次元空間の次元数　Number of dimensions in latent space </h2>

本研究ではVAEにより次元削減をしており，潜在空間は多次元正規分布を仮定している．<br>
その低次元空間の次元数は <strong>1,400</strong> 次元としている．


入力画像の3次元脳MR画像は約 <strong>500万次元</strong> であり，それを <strong>1,400</strong> 次元にまで圧縮している．



<h2> Reference </h2>
Daniel, Tal, and Aviv Tamar. "Soft-IntroVAE: Analyzing and improving the introspective variational autoencoder." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.


<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@InProceedings{Daniel_2021_CVPR,
author    = {Daniel, Tal and Tamar, Aviv},
title     = {Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2021},
pages     = {4391-4400}
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@InProceedings{Daniel_2021_CVPR,
author    = {Daniel, Tal and Tamar, Aviv},
title     = {Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2021},
pages     = {4391-4400}" tabindex="0" role="button" style="display: inherit;">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  
  
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

<br>
2023年第85回 情報処理学会 全国大会(2023/03/02～04) で発表
<br>
情報処理学会(https://www.ipsj.or.jp/index.html)
