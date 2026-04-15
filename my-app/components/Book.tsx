import React, { useState , useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import Image from 'next/image'
import { motion, AnimatePresence } from 'framer-motion'
import PeekingCharacters from './PeekingCharacters'


export default function Book({ storyData }: { storyData: any }) {
  const [currentPage, setCurrentPage] = useState(0)
  const [isFlipping, setIsFlipping] = useState(false)
  const [flipDirection, setFlipDirection] = useState<'left' | 'right'>('right')

  console.log('Story data in Book component:', storyData);

  // ...existing code...
const storyText = typeof storyData === "string" ? storyData : (storyData?.response ?? "");
const storyParagraphs = storyText.split('\n\n').filter((p: string) => p.trim());
// Ensure we have at least 4 pages (one for each generated image)
const pages = storyParagraphs.map((paragraph: string, index: number) => ({
  content: paragraph.trim(),
  image: `/Images/Image${index + 1}.png?timestamp=${new Date().getTime()}`,
}));
// ...existing code...
  const stopReading = () => {
    window.speechSynthesis.cancel();
  };
const handleReadOutLoud = () => {
  console.log('Handle Read Out Loud invoked.');
  if (pages[currentPage]?.content) {
    console.log('Reading content:', pages[currentPage].content);
    speakText(pages[currentPage].content);
  } else {
    console.warn('No content available to read.');
  }
};

const speakText = (text: string) => {
  if (!text) return;
  console.log('Speaking text:', text);
  window.speechSynthesis.cancel(); // Cancel ongoing speech synthesis
  const speech = new SpeechSynthesisUtterance(text);
  speech.lang = 'en-US';
  speech.rate = 1;
  window.speechSynthesis.speak(speech);
};

    // Speak the current page's content when it changes
    useEffect(() => {
      // Stop current speech
      window.speechSynthesis.cancel();
  
      // Speak the current page's content if available
      if (pages[currentPage]?.content) {
        speakText(pages[currentPage].content);
      }
    }, [currentPage, pages]);

  const turnPage = (direction: 'left' | 'right') => {
    if (isFlipping) return
    setIsFlipping(true)
    setFlipDirection(direction)

    // Fix for turning pages and preventing skipping or infinite turning
    const newPage = direction === 'left' ? Math.max(0, currentPage - 1) : Math.min(pages.length - 1, currentPage + 1)
    setTimeout(() => {
      setCurrentPage(newPage)
      setIsFlipping(false)
    }, 600)
  }

  return (
    <>
      <style jsx global>{`
        .book-wrapper {
          perspective: 1500px;
        }
        .book {
          position: relative;
          transform-style: preserve-3d;
          
        }
        .book-page {
          position: absolute;
          top: 0;
          width: 50%;
          height: 100%;
          transform-origin: center;
          transform-style: preserve-3d;
          transition: transform 0.6s ease;
        }
        .book-page-content {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: #fff;
          backface-visibility: hidden;
          padding: 20px;
          box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        }
        .book-page-back {
          transform: rotateY(180deg);
        }
        .page-fold {
          position: absolute;
          top: 0;
          right: 0;
          bottom: 0;
          width: 50%;
          background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.5));
          transform-origin: left;
          transition: transform 0.6s ease;
        }
      `}</style>
    <div className="flex flex-col items-center gap-4">
      <div className="flex items-center gap-8">
      <button
          onClick={() => turnPage('left')}
          disabled={currentPage === 0 || isFlipping}
          className="text-blue-600 hover:text-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label="Previous page"
        >
          <ChevronLeft size={40} />
        </button>

        <div className="book-wrapper">
          <div className="book w-[800px] h-[500px] bg-yellow-100 rounded-lg shadow-2xl overflow-hidden relative">
            <PeekingCharacters />
            <div className="absolute left-0 w-[400px] h-full bg-white p-8">
              <div className="prose prose-lg">
              <p className="text-lg font-serif text-gray-800 leading-relaxed">
                {pages[currentPage]?.content || "No content available"}
              </p>
              </div>
            </div>
            {/* Right page */}
            <div className="absolute left-[400px] w-[400px] h-full bg-white p-8">
              <div className="w-full h-full relative">
                <Image
                  src={pages[currentPage]?.image || '/Images/placeholder.svg'}
                  alt="Story illustration"
                  fill
                  unoptimized
                  className="object-cover rounded-lg"
                />
              </div>
            </div>

            {/* Animated flipping page */}
            <AnimatePresence>
              {isFlipping && (
                <motion.div
                  className="book-page"
                  initial={{ 
                    rotateY: flipDirection === 'right' ? 0 : -180,
                    x: flipDirection === 'right' ? '100%' : '0%'
                  }}
                  animate={{ 
                    rotateY: flipDirection === 'right' ? -180 : 0,
                    x: flipDirection === 'right' ? '0%' : '100%'
                  }}
                  exit={{ 
                    rotateY: flipDirection === 'right' ? -180 : 0,
                    x: flipDirection === 'right' ? '0%' : '100%'
                  }}
                  transition={{ duration: 0.6, ease: "easeInOut" }}
                >
                  <div className="book-page-content">
                    {flipDirection === 'right' ? (
                      <div className="prose prose-lg">
                        <p>{pages[currentPage + 1]?.content || "The End"}</p>
                      </div>
                    ) : (
                      <div className="w-full h-full relative">
                        <Image
                          src={pages[currentPage]?.image || '/Images/placeholder.svg'}
                          alt="Story illustration"
                          fill
                          unoptimized
                          className="object-cover rounded-lg"
                        />
                      </div>
                    )}
                    <motion.div 
                      className="page-fold"
                      initial={{ transform: 'scaleX(0)' }}
                      animate={{ transform: 'scaleX(1)' }}
                      exit={{ transform: 'scaleX(0)' }}
                      transition={{ duration: 0.6, ease: "easeInOut" }}
                    />
                  </div>
                  <div className="book-page-content book-page-back">
                    {flipDirection === 'right' ? (
                      <div className="w-full h-full relative">
                        <Image
                          src={pages[currentPage + 1]?.image || '/Images/placeholder.svg'}
                          alt="Story illustration"
                          fill
                          unoptimized
                          className="object-cover rounded-lg"
                        />
                      </div>
                    ) : (
                      <div className="prose prose-lg">
                        <p>{pages[currentPage]?.content}</p>
                      </div>
                    )}
                    <motion.div 
                      className="page-fold"
                      initial={{ transform: 'scaleX(1)' }}
                      animate={{ transform: 'scaleX(0)' }}
                      exit={{ transform: 'scaleX(1)' }}
                      transition={{ duration: 0.6, ease: "easeInOut" }}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <button
          onClick={() => turnPage('right')}
          disabled={currentPage >= pages.length - 1 || isFlipping}
          className="text-blue-600 hover:text-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label="Next page"
        >
          <ChevronRight size={40} />
        </button>
      </div>
      
      {/* Read out loud button */}
      <button
        onClick={handleReadOutLoud}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 transition"
      >
        Read Out Loud
      </button>
      

      {/* Stop button */}
      <button
        onClick={stopReading}
        className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-700 transition"
      >
        Stop Read Aloud
      </button>
    </div>
    </>
  );
}