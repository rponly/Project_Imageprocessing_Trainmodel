import React from 'react'

import NavigationBar from './NavigationBar'
import Dashboard from './Dashboard'

export class Main extends React.Component {
    render() {
        return (
            <React.Fragment>
                <NavigationBar />
                <Dashboard />
            </React.Fragment>
        )
    }
}

export default Main
