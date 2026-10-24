0:00 Have you ever wondered what would happen
0:01 if you converted an Excel spreadsheet to
0:03 an audio file or an MP4 video to a PDF
0:07 document? Or maybe there's some other
0:09 exotic combination worth trying out. But
0:11 I guess we'll never know because most
0:14 online file converters only let you
0:16 select formats in the same medium.
0:18 Images to [music] images, videos to
0:20 videos, text documents to fancier text
0:22 documents, and so on. What if there was
0:25 a service that could natively convert a
0:28 portal to demo file to a flash
0:30 animation, for example, and did [music]
0:32 so completely locally? All of these
0:35 online tools upload your files to a
0:37 server, which does all the processing.
0:39 I'm sure you can imagine why that's
0:40 terrible for privacy. In fact, I cannot
0:43 stress this enough. Uploading personal
0:45 files to a random internet server is one
0:48 of the most careless things that we've
0:49 normalized. So, I'm imagining a website
0:52 that can convert anything to anything
0:55 and locally in your own browser. These
0:59 are the main two goals that I set out
1:00 with to build my own file converter. If
1:02 you don't care about how I did it,
1:04 here's the link. You can go and try it
1:06 for yourself. Though, bit of a heads up,
1:08 there is a reason why no one has done
1:10 this before, and that's what I'll be
1:11 talking about in this video.
1:13 Specifically, the elephant in the room
1:15 is no doubt the promise of converting
1:17 between two completely different
1:19 mediums. Like, what would it even mean
1:21 to convert an MP4 video to a PDF? Well,
1:24 as you might already know, videos
1:27 consist of multiple sequential frames,
1:29 and a PDF document consists of multiple
1:31 sequential pages. When you frame it like
1:34 that, the process becomes pretty clear.
1:36 You extract the video's frames into a
1:38 series of images and then build a PDF
1:40 file with pages containing those images.
1:43 But here I want you to notice that we're
1:45 actually dealing with two problems at
1:48 once. Sure, we have to convert between
1:50 file formats like MP4, PNG, and PDF, but
1:53 we also have to convert between
1:55 different forms of media like digital
1:57 [music] images to document pages. For
2:00 videos and documents, this is pretty
2:02 straightforward. But things do get a bit
2:03 more complicated when you're converting
2:05 images to audio, for example. Even
2:07 though you might support the JPEG format
2:09 and the MP3 format, that doesn't make it
2:12 any easier to figure out how an image
2:14 becomes sound. Let's start with the
2:17 easier problem for now, which is that we
2:19 need to support as many file formats as
2:22 possible. Now, remember, this is
2:24 supposed to be a website that does all
2:26 of the computation on the client's
2:28 computer. That makes things a bit more
2:31 challenging than usual because it means
2:32 we can't just spin up VC to decode a
2:34 video for example. The tools we use have
2:37 to be compatible with a web browser.
2:40 Take audiovisisual media as an example.
2:42 The most popular utilities for such
2:44 formats are ffmpeg for video and audio
2:47 and image magic for images. If you've
2:49 never heard of them, you might be
2:51 surprised to find that they're the
2:52 backbone of many media focused services.
2:54 [music]
2:55 Even YouTube uses FFmpeg for transcoding
2:58 video, at least sometimes. Nowadays,
3:01 they do of course have proprietary tech
3:03 to do the more common stuff. While Image
3:05 Magic and FFmpeg are usually native
3:07 computer programs meant to run on a
3:09 desktop PC or a server, people have
3:12 ported them to Web Assembly, a
3:14 technology that makes it possible to run
3:16 them in a browser. This alone is a great
3:19 starting point. Just these two would
3:21 already make for a competent file
3:23 converter for images and videos, even
3:25 supporting some truly exotic formats.
3:28 FFmpeg can even read which is the
3:30 video format used in most Valve games.
3:33 But I, of course, have no plans of
3:35 stopping here. Let's also not forget
3:38 that the average browser already has
3:40 some built-in media capabilities. I
3:42 mean, how else would it decode the video
3:44 you're watching? Of course, in the
3:46 majority of cases, our specialized tools
3:48 do a much better job, but still the
3:50 built-in stuff can be useful sometimes.
3:53 For example, the web assembly port of
3:55 image magic doesn't come with support
3:57 for WEBP images. However, most browsers
4:01 can handle WEBP. So, I wrote a little
4:04 script that renders the WEBP on an
4:06 invisible canvas and then effectively
4:08 screenshots that to turn it into a PNG.
4:11 That PNG can then be fed directly into
4:14 Image Magic. So, I've basically added
4:16 support for an extra format with no
4:19 additional cost. Another thing that
4:22 browsers are good at is rendering
4:23 documents. I mean, think about it. Why
4:26 would I include a separate tool for
4:28 parsing web pages or vector graphics if
4:30 that's literally what your browser does
4:33 all the time? I can take the contents of
4:35 an HTML document, embed it in an SVG
4:38 image using the foreign object tag,
4:40 [music] and then just like before, put
4:42 it on an invisible canvas and screenshot
4:45 it into an image. [music]
4:46 But of course, when I say document, you
4:49 probably aren't thinking of a web page.
4:52 You are probably thinking of Office
4:54 [music] documents like those of
4:55 Microsoft Word, PowerPoint, Excel, and
4:58 so on. Unfortunately, [music] there
5:00 doesn't seem to be an all-in-one
5:02 solution for parsing these on a browser.
5:04 Sure, there are one-off tools
5:06 specialized in handling one format at a
5:08 time, and there are projects that glue
5:10 them together, but that quickly becomes
5:13 a bloated mess. On its own, this could
5:15 have been brushed aside as an
5:16 inconvenience, but that's still just the
5:18 beginning. While there are options for
5:20 all the major Microsoft document
5:22 formats, there aren't as many for the
5:24 open document alternatives. And
5:27 concerning the tools that do exist, the
5:29 most popular Word document to HTML
5:32 parser, Mammoth, converts a document
5:34 like this to this. It's not hard to spot
5:38 [music] the difference. And sure, I
5:40 could patch it to add support for
5:42 formatting, but Mammoth was never
5:44 designed to do that. Its purpose is just
5:46 to translate the document structure, and
5:48 I'd be fighting the tool if I wanted
5:50 authentic output. [music] The only other
5:52 reasonable alternative seemed to be
5:54 including yet another commonly used
5:56 open- source media parser as a web
5:58 assembly program. However, that would
6:01 likely mean bundling Libra Office.
6:03 [music] And as far as I can tell, this
6:05 one's not as mature as the rest. It
6:08 probably would give accurate results,
6:09 but I was more worried about the size
6:11 and loading times. [music] So,
6:14 naturally, when the tools you want don't
6:16 seem to exist, you go ahead and make
6:19 your own.
6:20 Yeah, I took a little bit of a detour to
6:23 work on a standalone office to HTML
6:26 parser in pure JavaScript. In
6:28 retrospect, this was probably a bad use
6:31 of my time. There are three major steps
6:34 to this. Extracting the zip, parsing the
6:36 XML, and rendering the document itself.
6:38 The first one is exactly what it sounds
6:40 like. A modern Microsoft Office or Open
6:42 Document file is just a renamed ZIP
6:45 file. Extracting zip files is a fairly
6:48 common task, so I didn't really feel
6:50 like reinventing the wheel on this one.
6:52 That said, the existing libraries come
6:54 with a bunch of stuff that I do not
6:56 need, like support for creating zip
6:58 files, which I never plan to do, and
7:00 support for decompression algorithms,
7:02 which I will never use. So instead, I
7:05 got Claude to implement the bare minimum
7:07 so that I could focus more on the
7:08 important aspects like parsing XML. XML
7:12 stands for extensible markup language
7:14 and it's basically a somewhat human
7:16 readable data structure used for all
7:18 sorts of things. When you extract a Word
7:20 document, these are the files you'll
7:22 find hidden within. They are what
7:24 describe the document's entire layout
7:26 and formatting. [music]
7:28 Now, if you've ever seen HTML, you might
7:30 notice that XML looks very similar. Most
7:33 browsers actually do support parsing XML
7:35 data, but I wanted to make sure that
7:37 this code can be used outside of a
7:39 browser, too. Luckily, writing an XML
7:42 parser is not at all difficult to do.
7:44 [music] My implementation came in at
7:46 about 70 lines of code. And while it
7:48 probably won't work for everything,
7:49 [music] it serves its purpose in my
7:51 specific use case. Now, all that's left
7:54 is to parse the actual document. I will
7:56 say the open document stuff is so much
7:59 nicer to work with than Microsoft's
8:01 format. It often translates directly to
8:04 HTML with very little need for
8:06 additional logic and [music] the
8:08 components are directly reusable. You
8:11 can pretty much just throw a
8:12 presentation into the text document
8:14 parser and it'll almost kind of work.
8:17 Open document files also come with a
8:19 thumbnail image. So if my parser ever
8:21 crashes on something, it can at least
8:23 still render a blurry copy of the first
8:26 page. Anyway, here are the results of my
8:28 parser compared to the source document.
8:31 And here they are compared to Mammoth's
8:33 output. To be clear, I'm not throwing
8:36 shade at Mammoth. It's just not the
8:38 right tool here. Of course, my tool also
8:40 supports parsing PowerPoint and Excel
8:42 documents to a limited extent. In
8:45 retrospect, I'm sure that forking and
8:47 skimming Libra Office would have been
8:48 the correct approach, and my makeshift
8:50 custom tool likely doesn't have the
8:52 stability of all the others. But hey, at
8:54 least it supports formatting. I'm
8:57 definitely way past recovery from the
8:59 Sunkost fallacy at this point, so I kind
9:01 of have to use this even if I think
9:03 Libra Office would be a better fit. As
9:06 you can probably tell by now, this is
9:08 not a serious project. This is just me
9:11 goofing around with a silly idea, hoping
9:13 to make something mildly practical out
9:15 of it. That said, if you do genuinely
9:18 want to improve either of these tools,
9:20 the GitHub links are in the description.
9:22 And also in the description is a link to the GitHub repositories.
10:29 Let's take a break from the file format
10:31 stuff for a moment to talk about the
10:33 second major problem, converting between
10:36 mediums. For visual stuff like images,
10:39 videos, documents, animations, fonts,
10:41 and so on, the answer seems to be pretty
10:44 obvious. Just convert them to an image.
10:47 Sure, you might lose some quality and
10:49 details along the way, particularly when
10:51 dealing with vector graphics, but the
10:53 result will likely be more or less what
10:55 you expect. Raster images are a good
10:57 enough common factor for lots of
10:59 different files. But what about audio?
11:03 Sound doesn't convert to an image in any
11:06 obvious way. Some audio files may embed
11:08 album art which you could extract, but
11:10 that's not really what we're talking
11:12 about, is it? We need to find a way to
11:14 represent sound visually right away. You
11:18 might propose a couple of naive ways of
11:20 going about this. For example, you could
11:22 dump the audio data straight into the
11:24 pixel data, but that would just leave
11:25 you with a bunch of incoherent noise.
11:28 You might also think to simply draw the
11:30 audio waveform, but that wouldn't
11:32 explain how images convert back to
11:35 audio. A waveform is one-dimensional. At
11:38 any point along the x-axis, there's only
11:40 one corresponding point on the yaxis.
11:43 This is simply not true for an arbitrary
11:46 image. How are you going to get a wave
11:48 from this? No, I want a solution that
11:51 creates a semantically significant image
11:53 [music] from the input audio and can
11:56 also produce a semantically significant
11:58 audio file from any input image. The
12:01 solutions we've discussed so far dabble
12:03 in the spatial domain. But for this,
12:06 we'll have to dive into the frequency
12:07 domain. You might remember from physics
12:10 class that all sound is just a wave. Any
12:13 regular wave can be described using just
12:15 a few parameters. the frequency, which
12:17 is how close the peaks are from each
12:19 other horizontally. The amplitude, which
12:21 is the distance between the highest and
12:23 lowest point, and the phase, which is
12:25 the offset from the start of the wave.
12:27 Basically, you can think of these as
12:29 different ways to squish, stretch, and
12:31 move the wave around. Audio engineers
12:33 are likely boiling in rage right now due
12:35 to the oversimplification, but this
12:36 explanation will do for now. However, a
12:39 regular wave will only produce one
12:42 constant tone.
12:45 More complicated sounds have to layer
12:47 multiple waves on top of each other and
12:50 change them over time to produce a wave
12:52 form. That's ultimately all that an
12:54 audio file is. It's just the final wave
12:57 gained by summing all of its component
12:59 waves. So to get an additional dimension
13:02 for our image, we can split this
13:04 waveform back into its component waves
13:06 by using what's called a discrete forier
13:08 transform. This way, each point on the
13:11 x-axis now corresponds to multiple
13:13 points on the y-axis, just like pixels
13:16 in an image. To actually translate this
13:18 to pixels, we can order the waves by
13:20 frequency along the y-axis. Then, for
13:22 each fragment of the wave, the red and
13:24 green color channels encode that wave's
13:27 amplitude, while the blue color channel
13:29 encodes the phase. And just like that,
13:32 we've represented a waveform as a raster
13:34 image. Not only that, we can then
13:36 reverse this whole process to get back
13:38 the input audio file. Or because this is
13:41 just a grid of pixels, we can even
13:43 convert arbitrary images to audio,
13:45 though that probably won't sound very
13:47 good. Of course, I'm massively
13:49 oversimplifying this whole process. I'm
13:52 not at all an expert on audio
13:54 processing, so my implementation
13:55 introduces quite a bit of noise in
13:57 certain areas. If you think you can help
13:59 improve this, please do. Again, the
14:01 GitHub link is in the description.
14:04 With audio and visual media covered, you
14:06 might be tempted to say that that's it.
14:09 Computers only display pictures and play
14:11 sound. So surely that's all you need to
14:13 represent all digital media. But let's
14:16 look at this from a different
14:17 perspective. Audio, as we've already
14:20 discussed, is one-dimensional. Images
14:22 are two-dimensional. So that of course
14:24 raises the question, do we have any
14:27 three-dimensional media? The most
14:29 obvious answer comes in the form of 3D
14:31 models. I mean, think about it. This
14:33 raises the exact same problems that
14:35 audio did. [music] Representing a 3D
14:38 model as a 2D render loses a whole
14:40 dimension of information. For that same
14:43 reason, it's also not a reversible
14:45 process. My immediate reaction would be
14:47 to slice the model on every vertex along
14:49 some axis. You could then use a library
14:51 like 3JS to render the slices with an
14:53 isometric camera and a depthbased
14:55 shader. This would allow you to later
14:57 reconstruct the model from the slices
14:59 based on their shape and pixel
15:01 brightness. At least that's the idea.
15:04 This is definitely outside of my area of
15:06 expertise, and this project is already
15:08 dragging on longer than it should. If
15:10 you're up for the task, community
15:11 contributions are once again welcome
15:13 here. Thankfully, I don't think I can
15:16 come up with any examples of
15:17 four-dimensional media. So, well,
15:20 actually, no. 3D animations do count as
15:24 four-dimensional. I suppose you could
15:26 divide them into multiple output files
15:27 like we already do for videos, but
15:30 again, I'll just stick with two
15:31 dimensions for now. So, all that's left
15:34 is to figure out how exactly all of
15:37 these tools link together. Let's say you
15:39 have a tool that handles videos and a
15:41 tool that handles images. You've [music]
15:44 been tasked to convert an MP4 video into
15:46 a series of PNGs. Neither of these tools
15:49 can do that on its own, so you'll have
15:51 to find a way to link them together. To
15:54 do that, you can look at the list of
15:55 formats supported by both tools and find
15:58 one that they have in common. In this
16:01 case, both FFmpeg and Image Magic
16:03 natively support the GIF format. It's
16:05 not quite an image, but it's also not
16:07 quite a video. So, your MP4 can become a
16:10 GIF, and that GIF can then become PNG.
16:14 By this logic, we can algorithmically
16:16 determine whether any one tool is
16:18 compatible with another. You just need a
16:20 list of formats outputed by the first
16:22 tool and a list of formats accepted by
16:24 the second tool. If there's a match in
16:26 these lists, you've got a connection. So
16:29 then the big idea is to throw all of our
16:32 tools into one big pile and have the
16:34 program route through all of them to get
16:36 from point A to point B, chaining
16:38 different formats along the way. It's
16:41 also important to note that we don't
16:42 just want any path. We want the shortest
16:45 path with the fewest links to improve
16:47 speed and reduce quality loss. We also
16:50 want to weigh our choices towards more
16:52 mainstream and ideally lossless formats.
16:54 For that same reason, I think it helps
16:57 to visualize the different tools as
16:58 nodes in a treelike data structure. We
17:01 start at a root node and make a list of
17:03 all the tools compatible with our input.
17:06 Then we search each of these tools to
17:08 see if they're capable of outputting our
17:10 target format. [music] Or in other
17:11 words, we're checking if they're
17:13 connected to some target node. [music]
17:15 If they're not, we descend downwards and
17:18 repeat the first step to generate a new
17:20 list of child nodes. We again check if
17:23 any of these tools are capable of
17:24 outputting our target format. And if
17:26 not, we move to an adjacent node.
17:29 Repeating the same process. This is a
17:31 common tree search algorithm known as
17:33 breath first search. It guarantees that
17:36 whatever solution we find is the
17:38 shortest one in terms of number of
17:39 nodes. as we only descend downward when
17:42 we've exhausted all options available on
17:44 the current level. In order to bias the
17:46 search towards more favorable formats,
17:48 we can simply reorder the nodes
17:50 horizontally so that the better formats
17:52 are looked at first. And really, that's
17:55 all there is to it. There's a bit more
17:58 logic to make sure that we don't route
17:59 through the same format more than once
18:01 and each candidate path is tested for
18:03 errors as some routes tend to only be
18:04 compatible on paper and not in practice.
18:07 But this also means that to add a new
18:10 format, you just have to be able to
18:12 convert it to and from just one of the
18:15 already supported formats. The algorithm
18:17 will then automatically find its way to
18:19 everything else.
18:21 Still, the algorithm itself is really
18:24 not the hard part. The hard part is
18:26 getting the required information out of
18:28 all of these tools and standardizing it
18:31 so that it can be cross-co compared.
18:33 Most file converter programs don't
18:35 really keep a list of supported formats.
18:37 It's surprisingly difficult to
18:39 accurately label a file format. [music]
18:41 Take MP4 for example. It might seem like
18:44 thempp4 file extension would be a good
18:47 enough label, but no, far from it. MP4
18:51 is really just a container for a range
18:54 of video and audio formats. [music]
18:56 It might not matter much for the end
18:58 user, but it does matter for the program
19:00 doing the conversion. Is this an H.264?
19:02 H.264 codec, H.265, AV1 is the audio AAC
19:06 or maybe Opus. This is actually the
19:08 reason why some MP4 files embed in
19:11 Discord while others don't. So, of
19:14 course, FFmpeg lists MP4 in a group of a
19:18 whole bunch of other containers. [music]
19:20 PNG has a similar issue. Yes, most of
19:23 the time thePNG extension is sufficient,
19:26 but not always. Animated PGs are rare,
19:30 but they do exist, and they often use
19:32 the same.png extension. Getting this
19:35 wrong could mean treating a video file
19:37 as an image file, which is obviously not
19:40 what we want. Other files have the
19:42 opposite problem. [music] Take XML for
19:44 example. As previously discussed, this
19:46 is used internally in Office documents,
19:49 but it's also the backbone of SVG images
19:51 and RSS feeds. These are obviously
19:54 different mediums, but they all share
19:56 the same underlying format. [music]
19:59 In order to better categorize each file
20:01 in my system, I label them all by their
20:03 mime type. MIME stands for multi-purpose
20:06 internet mail extensions. And [music] as
20:08 the name suggests, this was initially a
20:10 standard for email, but because of its
20:12 usefulness in categorizing file formats
20:14 by their semantic meaning, it's become a
20:16 sort of de facto standard for all kinds
20:19 of file related things. Now, admittedly,
20:21 this is still not a perfect system, but
20:24 it is far better than just using file
20:26 extensions. Still, there are many
20:27 formats that simply don't have a mime
20:30 type, and plenty of other formats have
20:32 too many mime types. And in rare cases,
20:35 one mime type can correspond to multiple
20:37 formats, as is the case with MP2 and
20:40 MP3. Regardless of which way you look at
20:43 it, there just is no universal way to
20:46 identify a file. We're working with what
20:48 we've got.
20:50 Finally, let's talk performance. [music]
20:53 A website like this is obviously going
20:55 to be very heavy. After all, we're
20:57 making the user download tens of
20:59 megabytes of binaries before they can
21:01 even do anything. But if you're just
21:03 here to convert a PNG to a JPEG and
21:05 don't really care about all that exotic
21:07 stuff, you probably don't need to load
21:09 the video or document tools. You don't
21:11 even need to load image magic really.
21:13 Instead, we can simply cache the lists
21:15 of supported formats and only load the
21:18 actual programs when they're being used.
21:21 Now, as you can imagine, the silliness
21:23 of this project is defined solely by the
21:25 formats that it supports. Sure, you can
21:28 transcode your videos to MP4 or convert
21:30 your images to PNG, but that's not as
21:32 fun as a 20page PDF document of the
21:35 Valve intro. Also, that example from the
21:37 start of the video wasn't a joke. This
21:39 thing really does support rendering
21:41 Portal 2 replay files. I know that
21:43 absolutely nobody will ever use that
21:45 feature, but I had to mention it here so
21:47 that the time spent adding it feels less
21:49 like a waste. In an ideal world, this
21:51 would support every file format to some
21:54 degree. But of course, that's an
21:56 unrealistically time-consuming task.
21:58 That's why, once again, I invite you to
22:01 contribute [music] if you can. All it
22:03 takes to add support for a new format is
22:05 to write a bit of code that converts it
22:07 to something that we already support.
22:09 [music] From there, the algorithm can
22:10 take it away and route you to literally
22:13 any other format on the list
22:14 automatically. So, as I said before, the
22:18 links to this project as well as the
22:19 GitHub repositories are in the
22:21 description. Thanks for [music]
22:23 watching.