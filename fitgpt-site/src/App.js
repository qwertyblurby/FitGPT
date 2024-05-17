import './App.css';
import GoogleFontLoader from 'react-google-font-loader';
import SectionWrapper from './SectionWrapper';
import Header from './Header';
import SectionHero from './SectionHero';
import SectionCaps from './SectionCaps';
import SectionResults from './SectionResults';
import SectionTech from './SectionTech';
import SectionAbout from './SectionAbout';
import Footer from './Footer';

function App() {
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
						<SectionHero />
					</SectionWrapper>
					
					{/* Capabilities */}
					<SectionWrapper>
						<SectionCaps />
					</SectionWrapper>
					
					{/* Demo results */}
					<SectionWrapper>
						<SectionResults />
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
