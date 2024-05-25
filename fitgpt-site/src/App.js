import { useState } from 'react';
import './App.css';
import GoogleFontLoader from 'react-google-font-loader';
import SectionWrapper from './SectionWrapper';
import Header from './Header';
import SectionHero from './SectionHero';
import SectionResults from './SectionResults';
import SectionCaps from './SectionCaps';
import SectionRecs from './SectionRecs';
import SectionTech from './SectionTech';
import SectionAbout from './SectionAbout';
import Footer from './Footer';

function App() {
	const [showResults, setShowResults] = useState(false);
	const [results, setResults] = useState(null);
	const [uploadedImage, setUploadedImage] = useState(null);
	
	const onUpload = async (event) => {
		try {
			setResults(null);
			setUploadedImage(null);
			const fileInput = event.target;
			const file = fileInput.files[0];
			if (file) {
				const formData = new FormData();
				formData.append('file', file);
				setShowResults(true);
				const response = await fetch('http://localhost:5000/upload', {
					method: 'POST',
					body: formData
				});
				
				if (!response.ok) {
					throw new Error("Network response was not ok");
				}
				
				const data = await response.json();
				const { output } = data;
				setResults(output);
				setUploadedImage(URL.createObjectURL(file));
				console.log("response received");
			} else {
				throw new Error("File not found");
			}
		} catch (error) {
			console.error("Error: ", error);
		};
	};
	
	
	return (
		<>
			<GoogleFontLoader
				fonts={[
					{
						font: "Chivo",
						weights: [400, 700],
					},
					{
						font: "Rubik",
						weights: [400, 500, 700],
					}
				]}
			/>
			
			<div className="App">
				<Header />
				
				<main>
					{/* Title and image */}
					<SectionWrapper>
						<SectionHero onUpload={onUpload}/>
					</SectionWrapper>
					
					{/* Results */}
					{showResults && (
						<SectionWrapper>
							<SectionResults results={results} uploadedImage={uploadedImage} />
						</SectionWrapper>
					)}
					
					{/* Capabilities */}
					<SectionWrapper>
						<SectionCaps />
					</SectionWrapper>
					
					{/* Demo recommendations */}
					<SectionWrapper>
						<SectionRecs />
					</SectionWrapper>
					
					{/* Technical details */}
					<SectionWrapper>
						<SectionTech />
					</SectionWrapper>
					
					{/* About us */}
					<SectionWrapper>
						<SectionAbout />
					</SectionWrapper>
				</main>
				
				<Footer />
			</div>
		</>
	);
}

export default App;
