# Çalışma Süresi (Uptime) Uygulamalarında Çok Kipli Erken Uyarı ve Dinamik Kapılı Füzyon ile Anomali Tespiti

**Ders**: YZM5110 - Araştırma Projesi (Çok Modlu / Çok Kipli Öğrenme)  
**Yazar**: A. Musab Arı  
**Kurum**: Yıldız Teknik Üniversitesi, Fen Bilimleri Enstitüsü, Bilgisayar Mühendisliği Anabilim Dalı  
**GitHub Depo Bağlantısı**: [https://github.com/amusabari-kion/Musab-Uptime_Multimodal](https://github.com/amusabari-kion/Musab-Uptime_Multimodal)  
**Tarih**: Kasım 2026  

---

## 1. Araştırma Sorusu, Kapsam, Motivasyon ve Özgün Katkılar

### 1.1 Araştırma Sorusu ve Kapsam
Yapısal olmayan metin tabanlı sistem günlükleri (loglar) ile sürekli zaman serisi telemetri metriklerinin çapraz kipli (cross-modal) ortak temsili, bulut sunucu çalışma süresi (uptime) izleme süreçlerinde tek kipli sistemlere kıyasla daha erken ve güvenilir anomali tespiti sağlayabilir mi? Kritik günlük servislerinin çöktüğü (örneğin syslog çökmesi) felaket senaryolarında bu çok kipli mimari nasıl dayanıklı kılınabilir?

Bu projenin kapsamı, %99.99 erişilebilirlik (yüksek uptime) garantisi gerektiren dağıtık sunucu altyapılarıdır. Hedef çalışma ortamında iki paralel veri akışı üretilmektedir: (i) periyodik olarak toplanan sayısal telemetri metrikleri (CPU, bellek, disk G/Ç, ağ trafiği, sunucu yükü ve sıcaklık) ve (ii) işletim sistemi ve konteyner servisleri tarafından olay güdümlü olarak oluşturulan metin tabanlı günlük kayıtları.

### 1.2 Motivasyon
Modern mikroservis ve bulut mimarilerinde meydana gelen karmaşık arıza kademelerinde, tek bir veri kipine dayanan izleme araçları ciddi kör noktalara sahiptir:
- **Yalnızca telemetriye dayalı izleme**: Zaman serisi metrikleri, donanım kaynaklarının tükendiğini veya ağ veriminin çöktüğünü gösterir; ancak kök neden açıklaması sunamaz. Örneğin, yüksek CPU kullanımı sonsuz bir iş parçacığı kilitlenmesinden mi yoksa meşru bir toplu işlemden mi kaynaklanmaktadır, ayırt edilemez.
- **Yalnızca günlüklere dayalı izleme**: Metin tabanlı sistem günlükleri, yazılım istisnaları ve hata kodlarına dair zengin anlamsal ipuçları taşır; fakat belirgin bir hata günlüğe yazılana kadar fiziksel kaynak aşınmasını fark edemez. Daha da kritiği, bellek tükenmesi (OOM kernel panic) veya ağ bölünmesi gibi büyük çöküşlerde günlük toplama servisleri (syslog, Fluentd vb.) bizzat çöker ya da tampon bellek taşması nedeniyle paketleri sessizce düşürür.

Bu iki modalitenin çapraz kipli füzyon ile birleştirilmesi karşılıklı bir doğrulama ve anlamsal zemin sağlar: metin günlükleri sayısal sıçramaları anlamlandırırken, telemetri eğilimleri günlük uyarılarının ciddiyetini derecelendirir.

### 1.3 Temel Katkılar
1. **MultiMon Temel Modelinin Yeniden Üretimi**: 1D-CNN telemetri kodlayıcısı ile Transformer günlük kodlayıcısını çapraz kipli dikkat (Cross-Modal Attention) ile birleştiren MultiMon temel mimarisi, halka açık Server Machine Dataset (SMD) veri setine uyarlanmış ve kısıtlı mini-yığın (batch size = 32) koşullarında başarıyla yeniden üretilmiştir.
2. **Dinamik Kapılı Füzyon (Dynamic Gated Fusion - DGF) Katmanı**: Standart çapraz dikkat mekanizmalarının eksik/bozuk kipler altındaki yapısal zaafı tespit edilmiş; günlük servisleri çöktüğünde bozulan metin akışını telemetriden yalıtan ve kip güvenilirliğini dinamik olarak kestiren bir kapı mekanizması önerilmiştir.
3. **Dayanıklılık ve Kaynak Ödünleşiminin Kapsamlı Ölçümü**: Temiz veri ve %50 günlük kaybı içeren bozuk veri koşullarında yapılan deneylerde; temel MultiMon modelinin F1 skoru %89.0'dan %45.0'a çökerken, önerilen GatedMultiMon modeli %78.0 F1 başarımı sağlamıştır (+%33.0 geri kazanım). Öte yandan dürüst bir hata analizi yapılarak, kapı katmanının getirdiği olumsuz yan etkiler (en yüksek GPU bellek tüketiminin 4.0 GB'tan 5.2 GB'a çıkması ve çıkarım gecikmesinin 14.2 ms'den 16.8 ms'ye yükselmesi) raporlanmıştır.

---

## 2. Literatür Arama Yaklaşımı ve Eleştirel Sentez

### 2.1 Literatür Arama Metodolojisi
Literatür taraması IEEE Xplore, ACM Digital Library, Google Scholar ve arXiv veritabanlarında gerçekleştirilmiştir. Arama sorgularında temel modalite terimleri `("log anomaly detection" OR "multivariate time series anomaly")`, birleştirme kavramları `("multimodal fusion" OR "cross-modal attention")` ve alan kısıtları `("cloud uptime" OR "AIOps" OR "server monitoring")` birlikte kullanılmıştır. Seçilen çalışmalar 2017–2023 yılları arasındaki çığır açıcı ve güncel makaleleri kapsamaktadır.

### 2.2 Literatür Karşılaştırma Matrisi

| Alan | 1. MultiMon (Wang vd., 2022) | 2. LogBERT (Guo vd., 2021) | 3. OmniAnomaly (Su vd., 2019) | 4. DeepLog (Du vd., 2017) | 5. Time-Series Transformer (Zhou vd., 2021) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Soru ve katkı** | Günlük ve telemetrinin ortak füzyonu; tek kipli bulut izleme eksikliklerini çözer. | Günlük anahtar dizilerine uygulanan öndenetimli maskeli dil modellemesi. | Çok değişkenli telemetri için düzlemsel normalleştirici akışlara sahip stokastik RNN. | Normal günlük geçişlerini ve parametre değerlerini modelleyen öncü LSTM ağı. | Zaman serisi karmaşıklığını $O(L^2)$'den $O(L \log L)$'ye düşüren ProbSparse dikkat mekanizması. |
| **Modaliteler ve kabuller** | Telemetri zaman serisi + sistem günlükleri; eşzamanlı ve tam veri akışı varsayar. | Yapılandırılmamış metin günlükleri; günlük servisinin kesintisiz çalıştığını varsayar. | Çok değişkenli sayısal zaman serisi (38 öznitelik); metrik erişilebilirliğini varsayar. | Yapısal olmayan sistem günlükleri; katı sıra düzeni ve şablon çıkarılabilirliği varsayar. | Uzun dizi sayısal zaman serisi; sürekli zamansal akış varsayar. |
| **Mekanizma ve amaç** | 1D-CNN + Transformer ve iki yönlü çapraz dikkat; ikili sınıflandırma kaybı. | BERT tarzı maskeli günlük belirteci tahmini ve hiperküre hacmi en aza indirme amacı. | VAE tabanlı yeniden yapılandırma olasılığı ve düzlemsel normalleştirici akış uzayı. | Çapraz Entropi ile sonraki belirteç tahmini; ilk $g$ adayın dışındaki sapmaları anomali sayar. | ProbSparse dikkatli üretken kod çözücü; MSE tahmin hatasını en aza indirir. |
| **Veri ve protokol** | Şirket içi özel bulut kümesi verisi; pencerelenmiş olay kayıtları üzerinde test edilmiştir. | Açık günlük veri setleri (HDFS, BGL, Thunderbird); sabit eğitim/test bölümleri. | 28 sunucu makinesini kapsayan Server Machine Dataset (SMD); Kesinlik, Duyarlılık, F1. | HDFS sistem günlükleri (11.1 milyon log); pencere tabanlı anomali tespiti. | Hava Durumu, Trafik, Elektrik ve SMD veri setleri; kayan tahmin değerlendirme protokolü. |
| **Temel modeller ve kanıt** | Bağımsız CNN (metrik) ve BERT (log) ile kıyaslanmış; +%8 F1 artışı kanıtlanmıştır. | DeepLog, PCA ve LogCluster ile kıyaslanmış; gürültülü günlüklerde üstünlük gösterilmiştir. | Bağımsız LSTM ve geleneksel İzolasyon Ormanları (Isolation Forest) ile kıyaslanmıştır. | Temel Bileşen Analizi (PCA) ve değişmez madenciliği yöntemleri ile kıyaslanmıştır. | Informer, standart Transformer, LSTNet ve ARIMA temel modelleriyle kıyaslanmıştır. |
| **Kaynaklar ve tekrarlanabilirlik** | Özel endüstri verisi; 4x Tesla V100 GPU; doğrudan tekrarlanabilirlik kısıtlıdır. | Kaynak kod açık; tek tüketici GPU'su (GTX 1080Ti); yüksek tekrarlanabilirlik. | SMD veri seti ve kod GitHub'da açık; standart iş istasyonunda tekrarlanabilir. | HDFS verisi açık; Drain ayrıştırıcı ön işlemesi gerektirir; kodlar yaygın mevcuttur. | Açık kaynaklı PyTorch deposu; tek GPU'lu standart iş istasyonunda çalıştırılabilir. |
| **Sınırlılıklar ve bağlantılar** | Günlük akışı kesildiğinde model çöker; kip güvenilirlik kapılaması yoktur. | Fiziksel metrik anomalilerine (bellek sızıntısı, aşırı ısınma) karşı tamamen kördür. | Uygulama düzeyindeki anlamsal hataları göremez; ani yük patlamalarında yanlış alarm üretir. | Yeni yazılım sürümlerinde ve şablon patlamalarında genelleyemez. | Çok uzun dizilerde seyreklik hileleri olmadan karesel hesaplama yükü oluşturur. |

### 2.3 Eleştirel Sentez ve Araştırma Boşluğu
Tek kipli modeller, gözlemledikleri modalitenin doğal sınırlarıyla kısıtlıdır. Günlük tabanlı modeller (DeepLog, LogBERT), sistem hata mesajı basamadan donduğunda başarısız olur. Telemetri modelleri (OmniAnomaly, Informer) ise varyans sıçramalarını yakalasa da teşhis derinliğinden yoksundur.

MultiMon çapraz dikkatin değerini kanıtlamış olsa da, en büyük zaafı girdilerin daima gürültüsüz ve eksiksiz olduğunu varsaymasıdır. Gerçek dünya bulut sistemlerinde günlük servisleri kırılgandır; bellek taşması veya ağ kesintilerinde syslog çöker. Standart çapraz dikkat mekanizması bu durumda boş vektörler üzerine dikkat ağırlığı normalize ederek temiz telemetri temsilini zehirler. Bu araştırma boşluğu, eksik kip durumunda telemetri sinyalini koruyan dinamik bir kapı mekanizmasının geliştirilmesini zorunlu kılmaktadır.

---

## 3. Seçilen Kavram ve Yöntemin Teknik Açıklaması

### 3.1 MultiMon Temel Mimarisi
MultiMon, zaman serisi telemetri verisi $X_{ts} \in \mathbb{R}^{B \times C \times L}$ ($B$: yığın boyutu, $C=8$: metrik sayısı, $L=15$: zaman penceresi) ile metin günlük dizilerini $X_{text} \in \mathbb{Z}^{B \times T}$ ($T=24$: belirteç uzunluğu) ortak bir temsilde birleştirir.

#### 3.1.1 Telemetri Metrik Kodlayıcısı (1D-CNN)
Zaman serisi girdisi iki ardışık 1D evrişim bloğundan geçer:
$$H_1 = \text{ReLU}\left(\text{BatchNorm}\left(\text{Conv1D}(X_{ts}, W_1)\right)\right)$$
$$H_{ts\_raw} = \text{ReLU}\left(\text{BatchNorm}\left(\text{Conv1D}(H_1, W_2)\right)\right)$$
Uyarlamalı ortalama havuzlama ile zaman boyutu indirgenerek yoğun telemetri vektörü elde edilir:
$$h_{ts} = \text{AdaptiveAvgPool1D}(H_{ts\_raw}) \in \mathbb{R}^{B \times d} \quad (d=128)$$

#### 3.1.2 Günlük Metin Kodlayıcısı (Transformer)
Günlük belirteçleri bir gömme matrisi $E \in \mathbb{R}^{V \times d}$ ile yoğun vektörlere dönüştürülür ve Transformer kodlayıcı katmanlarına iletilir:
$$H_{text\_seq} = \text{TransformerEncoder}\left(E(X_{text})\right) \in \mathbb{R}^{B \times T \times d}$$
Maskeli ortalama havuzlama ile özet metin vektörü $h_{text} \in \mathbb{R}^{B \times d}$ oluşturulur.

#### 3.1.3 Çapraz Kipli Dikkat (Cross-Modal Attention)
Kiplar arası etkileşimde telemetri Sorgu ($Q$), metin günlükleri ise Anahtar ($K$) ve Değer ($V$) rolü üstlenir:
$$Q = h_{ts} W_Q, \quad K = h_{text} W_K, \quad V = h_{text} W_V$$
$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times 1 \times 1}$$
$$h_{cross} = \text{LayerNorm}\left(h_{ts} + \text{Dropout}(A \cdot V)\right) \in \mathbb{R}^{B \times d}$$
Elde edilen birleşik vektör iki katmanlı bir MLP sınıflandırıcıya beslenerek anomali olasılığı hesaplanır.

---

## 4. Yeniden Üretim Hedefi, Gerçekleştirim ve Kaynaktan Sapmalar

### 4.1 Hedef İddia ve Beklenen Davranış
Kaynak makale, MultiMon modelinin çapraz dikkat sayesinde tek kipli modellere kıyasla üstün anomali tespit başarımı (%89–91 F1) elde ettiğini öne sürmektedir. Bu çalışmadaki amacımız, söz konusu çapraz kipli sinerjiyi doğrulamak ve günlükler eksildiğinde modelin nasıl tepki verdiğini incelemektir.

### 4.2 Yapılabilirlik İçin Yapılan Uyarlamalar ve Sapmalar
Orijinal MultiMon veri seti kapalı bir telekomünikasyon bulutuna ait olduğundan, çalışma akademik ve kamuya açık olan **Server Machine Dataset (SMD)** üzerine uyarlanmıştır:
1. **Veri Seti Uyarlaması**: 28 farklı sunucu makinesinden toplanan 38 boyutlu SMD telemetri metrikleri kullanılmış; buna ek olarak gerçek arıza dağılımlarını yansıtan eşzamanlı sistem log olayları sentezlenerek eşleştirilmiştir.
2. **Hesaplama Bütçesi ve Yığın Boyutu**: Orijinal çalışmada 4 adet Tesla V100 GPU üzerinde 128 yığın boyutu ile yapılan dağıtık eğitim; tek GPU/CPU ortamında yürütülebilmesi amacıyla 32 yığın boyutu, $d=128$ gizli katman genişliği ve 10 eğitim dönemi (epoch) ile sınırlandırılmıştır.
3. **Karşılaştırılabilirlik**: Yapılan bu mütevazı bütçe uyarlaması, temel model ile geliştirilmiş model arasındaki kontrollü başarım farkının adil bir zeminde kıyaslanmasını engellemez.

---

## 5. Önerilen İyileştirme ve Gerekçesi

### 5.1 Günlük Kayıplarında Temel Modelin Çöküş Nedeni
Sunucu üzerinde işletim sistemi kilitlendiğinde veya syslog servisi çöktüğünde hiçbir metin günlüğü üretilemez. Sayısal tensörlerde bu durum sıfırlanmış gömmeler veya dolgu (padding) belirteçleri olarak yansır. Ancak standart softmax operatörü gereği:
$$\sum_{j} \text{softmax}(S)_{ij} = 1.0$$
olmak zorundadır. Bu matematiksel zorunluluk, dikkat ağırlıklarının anlamsız boşluklar üzerine normalize olmasına ve $h_{cross}$ temsilinin bozulmasına yol açar. Sonuç olarak sağlam durumdaki telemetri sinyali de zehirlenir.

### 5.2 Dinamik Kapılı Füzyon (DGF) Katmanı
Bu yapısal zaafı ortadan kaldırmak için **Dinamik Kapılı Füzyon (Dynamic Gated Fusion - DGF)** katmanı önerilmiştir. Model, $h_{cross}$ vektörünü doğrudan sınıflandırıcıya iletmek yerine bir modalite güvenilirlik kapı vektörü $g \in [0, 1]^d$ hesaplar:
$$g = \sigma\left(W_{g2} \cdot \text{ReLU}\left(W_{g1} [h_{ts} \,\|\, h_{text}] + b_{g1}\right) + b_{g2}\right)$$
Nihai füzyon temsili, telemetri omurgası ile kapılanmış çapraz dikkati dinamik olarak dengeler:
$$h_{fused} = \text{LayerNorm}\left(W_{ts} h_{ts} + g \odot W_c h_{cross}\right)$$

Günlükler sağlamsa $g \approx 1$ olur ve çok kipli zengin bilgi korunur. Günlükler kaybolduğunda veya aşırı gürültülü olduğunda ise kapı sıfıra yaklaşır ($g \to 0$); böylece bozulan çapraz dikkat yolu devreden çıkarılarak modelin yalnızca güvenilir telemetri sinyaliyle hatasız sınıflandırma yapması sağlanır.

---

## 6. Deneysel Tasarım, Veri Setleri, Bölümler, Temel Modeller ve Metrikler

### 6.1 Veri Seti ve Kronolojik Bölümleme
Server Machine Dataset (SMD) üzerinde zaman serisi anomali tespitinde geleceğe sızıntıyı (lookahead bias) engellemek amacıyla rastgele karıştırma yapılmamış, katı bir kronolojik bölme uygulanmıştır:
- **Eğitim Kümesi (%70)**: İlk zaman penceresi; model parametrelerinin öğrenilmesinde kullanılır.
- **Doğrulama Kümesi (%10)**: Takip eden ara zaman penceresi; hiperparametre seçimi için ayrılmıştır.
- **Test Kümesi (%20)**: En son zaman penceresi; nihai model başarımı için test edilir.

### 6.2 Değerlendirme Koşulları
1. **Temiz Koşul (Clean)**: %100 eksiksiz eşleştirilmiş telemetri ve günlük akışı.
2. **Bozulmuş Koşul (Corrupted)**: Test aşamasında %50 oranında rastgele günlük kaybı; syslog servis çökmesi ve tampon bellek taşması simülasyonu.

### 6.3 Değerlendirme Metrikleri
- **Görev Kalitesi**: Kesinlik (Precision), Duyarlılık (Recall) ve F1 Skoru ($F_1$).
- **Hesaplama Yükü**: Ortalama çıkarım gecikmesi (örnek başına milisaniye - ms) ve En Yüksek GPU Bellek Tüketimi (Gigabayt - GB).

---

## 7. Sonuçlar, Kontroller, Belirsizlik ve Hata Analizi

### 7.1 Sayısal Deney Sonuçları

| Model Mimarisi | Değerlendirme Koşulu | Kesinlik | Duyarlılık | F1 Skoru | Gecikme (ms) | En Yüksek Bellek (GB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiMon (Temel Model)** | Temiz Günlükler (%0 kayıp) | **%89.5** | %88.5 | **%89.0** | **14.2** | **4.0** |
| **MultiMon (Temel Model)** | Bozulmuş Günlükler (%50 kayıp) | %78.2 | %32.1 | **%45.0** | 14.1 | 4.0 |
| **GatedMultiMon (Önerilen)** | Temiz Günlükler (%0 kayıp) | %88.2 | **%88.8** | **%88.5** | 16.8 | 5.2 |
| **GatedMultiMon (Önerilen)** | Bozulmuş Günlükler (%50 kayıp) | **%81.0** | **%75.4** | **%78.0** | 16.9 | 5.2 |

```text
F1 Skoru Karşılaştırması: Temiz vs. %50 Bozulmuş (Kayıp Günlükler)
===================================================================
MultiMon (Temiz)        : [████████████████████████████████████████████] %89.0
MultiMon (Bozulmuş)     : [███████████████████████                     ] %45.0 (-%44.0 çöküş)
GatedMultiMon (Temiz)   : [██████████████████████████████████████████  ] %88.5 (-%0.5 ödün)
GatedMultiMon (Bozulmuş): [███████████████████████████████████         ] %78.0 (+%33.0 toparlanma)
```

### 7.2 Hata ve İyileşme Analizi
Temiz ortamda temel MultiMon modeli %89.0 F1 başarımı gösterir. Ancak günlüklerin %50'si kesildiğinde, temel modelin duyarlılığı %88.5'ten %32.1'e düşer ve F1 skoru %45.0'a çakılır. İncelenen hata örneklerinde, dikkat matrisinin boş belirteçlere aşırı ağırlık vererek telemetri sıçramalarını gizlediği görülmüştür.

Önerilen GatedMultiMon modeli ise aynı %50 bozuk koşul altında %78.0 F1 skoru elde ederek temel modele göre +%33.0'lık devasa bir toparlanma sergilemiştir. Temiz verideki skorunun (%88.5) temel modelden yalnızca %0.5 düşük olması, kapı katmanının olağan çalışma durumunda neredeyse hiçbir başarım kaybı yaratmadığını kanıtlamaktadır.

### 7.3 Olumsuz Yan Etkilerin (Kaynak Yükünün) Dürüst Raporlanması
Dinamik kapılama mekanizmasının sunduğu yüksek dayanıklılık, ek donanım maliyetlerini beraberinde getirmiştir:
- **Bellek Tüketim Artışı**: En yüksek GPU bellek kullanımı 4.0 GB'tan 5.2 GB'a yükselmiştir (+%30.0). Bu artış, kapı projeksiyon matrisleri ve çift yönlü ara tensörlerin bellekte tutulmasından kaynaklanmaktadır.
- **Çıkarım Gecikmesi**: Örnek başına gecikme 14.2 ms'den 16.8 ms'ye çıkmıştır (+%18.3). Sigmoid aktivasyonları ve tensör çarpımları ek mikro-gecikmeler yaratmaktadır.

---

## 8. Sınırlılıklar, Sonuçlar ve Çözülmemiş Sorular

### 8.1 Sınırlılıklar
GatedMultiMon eksik günlük senaryolarında üstün dayanıklılık gösterse de, 5.2 GB bellek gereksinimi nedeniyle kaynakları son derece kısıtlı uç cihazlarda (edge IoT gateways) doğrudan çalıştırılmaya elverişli değildir. Ayrıca hem telemetrinin hem de günlüklerin aynı anda kesildiği uç senaryolarda modelin toparlanma yeteneği bulunmamaktadır.

### 8.2 Sonuçlar
Çok kipli öğrenme, bulut altyapılarında erken anomali tespiti için hayati avantajlar sunar. Ancak saf çapraz dikkat modelleri kip kayıplarında son derece kırılgandır. Dinamik kapılı füzyon katmanı, güvenilirlik tahmini yaparak telemetri omurgasını korumakta ve %50 günlük kaybında F1 skorunu %45.0'dan %78.0'a çıkarmaktadır.

### 8.3 Gelecek Çalışmalar ve Açık Sorular
1. Model parametrelerinin INT8 kuantizasyonu veya hafif bir öğrenci ağa damıtılması (knowledge distillation) ile bellek ayak izi 2 GB'ın altına indirilebilir mi?
2. Günlüklerin tamamen kaybolması yerine satırların yarıda kesildiği kısmi bozulma durumlarında kapılama hassasiyeti nasıl etkilenmektedir?
